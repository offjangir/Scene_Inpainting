import os, yaml, torch, random, math, numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

# ==== diffusers (Flux / ControlNet) ====
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
)
from diffusers.models import FluxControlNetModel, FluxTransformer2DModel
from diffusers.training_utils import compute_loss_weighting_for_sd3
from diffusers.utils.torch_utils import is_compiled_module

import wandb
import os
# Note: Cache directories are now set in ~/.bashrc


# ============================================================
# === Dataset (same layout you used) =========================
# ============================================================
class InpaintPairDataset(Dataset):
    """
    Dataset for training Flux ControlNet for 3D inpainting.
    - Conditioning: warped/masked view (image_cond.png) + mask (mask_cond.png)
    - Target: original clean view (image_target.png)
    """
    def __init__(self, root, size=1024,
                 image_suffix="image_target.png",      # original clean view (TARGET to generate)
                 cond_suffix="image_cond.png",          # warped/masked view (CONDITIONING input)
                 mask_suffix="mask_cond.png"):
        self.root = root
        self.size = size
        self.samples = []
        for scene_dir in sorted(os.listdir(root)):
            scene_path = os.path.join(root, scene_dir)
            if not os.path.isdir(scene_path):
                continue
            for pair_dir in sorted(os.listdir(scene_path)):
                pair_path = os.path.join(scene_path, pair_dir)
                if not os.path.isdir(pair_path):
                    continue
                img_path  = os.path.join(pair_path, image_suffix)
                cond_path = os.path.join(pair_path, cond_suffix)
                mask_path = os.path.join(pair_path, mask_suffix)
                if all(os.path.exists(p) for p in [img_path, cond_path, mask_path]):
                    self.samples.append({"image": img_path, "conditioning": cond_path, "mask": mask_path})
                else:
                    print(f"[⚠️] Skipping incomplete pair: {pair_path}")

        print(f"✅ Found {len(self.samples)} valid samples from {root}")

        self.tform_rgb = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.tform_mask = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Target: original clean view (what we want to generate)
        target_img = self.tform_rgb(Image.open(s["image"]).convert("RGB")).clamp(0,1)
        # Conditioning: warped/masked view (what we observe)
        warped_view = self.tform_rgb(Image.open(s["conditioning"]).convert("RGB")).clamp(0,1)
        mask = self.tform_mask(Image.open(s["mask"]).convert("L")).clamp(0,1)
        
        # Return separately for more flexible processing during training
        return {
            "image": target_img,
            "conditioning": warped_view,
            "mask": mask
        }


# ============================================================
# === Utils ==================================================
# ============================================================
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def encode_latents(pixels, vae: AutoencoderKL, dtype):
    # Flux uses (latent - shift) * scaling_factor
    with torch.no_grad():
        latents = vae.encode((pixels*2-1).to(vae.dtype)).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents.to(dtype)


def inspect_controlnet_cond_embedding(controlnet):
    """
    Helper function to inspect the controlnet_cond_embedding structure.
    
    The input to controlnet_cond_embedding is:
    - Shape: [B, C, H, W] where C is the number of conditioning channels
    - For inpainting: typically 4 channels (RGB image + mask)
    
    The flow is:
    1. controlnet_cond (input) -> input_hint_block (ControlNetConditioningEmbedding)
    2. input_hint_block.conv_in expects in_channels matching your control image channels
    3. input_hint_block processes through conv layers and outputs conditioning_embedding_channels
    4. Output is reshaped and passed to controlnet_x_embedder
    """
    print("\n" + "="*60)
    print("ControlNet Conditioning Embedding Inspection")
    print("="*60)
    
    if controlnet.input_hint_block is None:
        print("❌ No input_hint_block found (conditioning_embedding_channels is None)")
        return
    
    print(f"\n📥 INPUT to controlnet_cond_embedding:")
    print(f"   - Expected shape: [batch_size, input_channels, height, width]")
    print(f"   - Input channels (conv_in.in_channels): {controlnet.input_hint_block.conv_in.in_channels}")
    print(f"   - Kernel size: {controlnet.input_hint_block.conv_in.kernel_size}")
    print(f"   - Padding: {controlnet.input_hint_block.conv_in.padding}")
    
    print(f"\n🔄 PROCESSING (input_hint_block):")
    print(f"   - Type: ControlNetConditioningEmbedding")
    print(f"   - Block out channels: {controlnet.input_hint_block.blocks}")
    print(f"   - Number of blocks: {len(controlnet.input_hint_block.blocks) // 2}")
    
    print(f"\n📤 OUTPUT from controlnet_cond_embedding:")
    print(f"   - Output channels (conditioning_embedding_channels): {controlnet.config.conditioning_embedding_channels}")
    print(f"   - This goes to controlnet_x_embedder (Linear layer)")
    print(f"   - controlnet_x_embedder input dim: {controlnet.controlnet_x_embedder.in_features}")
    print(f"   - controlnet_x_embedder output dim: {controlnet.controlnet_x_embedder.out_features}")
    
    print("\n" + "="*60 + "\n")


def validate_and_visualize_flux(
    vae, transformer, controlnet, val_samples, device, epoch, output_dir,
    prompt_embeds, pooled_prompt_embeds, num_inference_steps=20, seed=42
):
    """
    Compute validation loss and generate visualizations for Flux
    
    Returns:
        tuple: (vis_dir, avg_val_loss)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    controlnet.eval()
    
    vis_dir = os.path.join(output_dir, f"epoch_{epoch}_vis")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"  📁 Saving visualizations to: {vis_dir}")
    
    # ===== PART 1: Compute Validation Loss =====
    print(f"  📊 Computing validation loss...")
    val_losses = []
    
    with torch.no_grad():
        for idx, sample in enumerate(val_samples):
            try:
                images = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
                conds = sample["conditioning"].unsqueeze(0).to(device)  # [1, 3, H, W]
                masks = sample["mask"].unsqueeze(0).to(device)  # [1, 1, H, W]
                
                # Whiten where mask=1 (same as SDXL)
                conds_white = torch.where(masks == 1, 1.0, conds)
                controls = torch.cat([conds_white, masks], dim=1)  # [1, 4, H, W]
                
                # Encode to latents
                x0 = encode_latents(images, vae, dtype=torch.float32)
                
                # Sample noise & time
                z = torch.randn_like(x0)
                t = torch.rand((x0.shape[0],), device=device)
                
                # Interpolate
                t_expanded = t.view(-1, 1, 1, 1)
                xt = (1.0 - t_expanded) * x0 + t_expanded * z
                u_t = z - x0
                
                # Weighting
                weights = compute_loss_weighting_for_sd3(t).view(-1, 1, 1, 1)
                guidance = torch.full((1,), 3.5, device=device, dtype=torch.float32)
                
                # Expand text embeddings
                text_embeds_batch = prompt_embeds[:1]
                pooled_embeds_batch = pooled_prompt_embeds[:1]
                
                # ControlNet
                ctrl_out = controlnet(
                    hidden_states=xt,
                    controlnet_cond=controls,
                    timestep=t / 1000.0,
                    guidance=guidance,
                    encoder_hidden_states=text_embeds_batch,
                    pooled_projections=pooled_embeds_batch,
                )
                
                # Transformer (move to GPU for forward pass)
                transformer.to(device)
                pred_v = transformer(
                    xt,
                    timestep=t / 1000.0,
                    guidance=guidance,
                    encoder_hidden_states=text_embeds_batch,
                    pooled_projections=pooled_embeds_batch,
                    block_controlnet_hidden_states=ctrl_out.block_controlnet_hidden_states
                    if hasattr(ctrl_out, "block_controlnet_hidden_states") else None,
                ).sample
                transformer.to("cpu")
                torch.cuda.empty_cache()
                
                # Compute loss
                loss = (weights * (pred_v - u_t).pow(2)).mean()
                val_losses.append(loss.item())
                
            except Exception as e:
                print(f"  ⚠️ Error computing validation loss for sample {idx}: {e}")
                continue
    
    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
    print(f"  ✅ Validation Loss: {avg_val_loss:.4f} (from {len(val_losses)} samples)")
    
    # ===== PART 2: Generate Visualizations =====
    print(f"  🎨 Generating visualizations...")
    images_to_log = []
    
    # For visualization, we'd need to implement full Flux sampling which is complex
    # For now, just visualize the inputs
    for idx, sample in enumerate(val_samples[:4]):  # Limit to 4 samples
        try:
            mask = sample["mask"].cpu().numpy()
            cond_image = sample["conditioning"].cpu().numpy()
            gt_image = sample["image"].cpu().numpy()
            
            # Whiten where mask=1
            mask_expanded = mask  # [1, H, W]
            cond_white = np.where(mask_expanded == 1, 1.0, cond_image)
            
            # Create visualization
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            
            # Mask
            axes[0].imshow(mask.squeeze(), cmap='gray')
            axes[0].set_title("Input Mask", fontsize=14)
            axes[0].axis('off')
            
            # Conditioning (with white holes)
            cond_np = cond_white.transpose(1, 2, 0)  # CHW -> HWC
            axes[1].imshow(np.clip(cond_np, 0, 1))
            axes[1].set_title("Input (Masked)", fontsize=14)
            axes[1].axis('off')
            
            # Placeholder for generated (would need full inference)
            axes[2].text(0.5, 0.5, 'Generated\n(Not implemented)', 
                        ha='center', va='center', fontsize=16)
            axes[2].set_title("Generated", fontsize=14)
            axes[2].axis('off')
            
            # Ground Truth
            gt_np = gt_image.transpose(1, 2, 0)
            axes[3].imshow(np.clip(gt_np, 0, 1))
            axes[3].set_title("Ground Truth", fontsize=14)
            axes[3].axis('off')
            
            plt.tight_layout()
            
            # Save
            save_path = os.path.join(vis_dir, f"sample_{idx}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            if os.path.exists(save_path):
                print(f"  ✓ Saved visualization: {save_path}")
                images_to_log.append(
                    wandb.Image(save_path, caption=f"Epoch {epoch} - Sample {idx}")
                )
        
        except Exception as e:
            print(f"  ⚠️ Error generating visualization {idx}: {e}")
            continue
    
    # Log to wandb
    if images_to_log:
        wandb.log({
            "validation/samples": images_to_log,
            "validation/loss": avg_val_loss,
            "validation/epoch": epoch
        })
        print(f"  📊 Logged {len(images_to_log)} images to W&B")
    
    controlnet.train()
    return vis_dir, avg_val_loss


# ============================================================
# === Training (Flux + ControlNet + Flow-Matching) ===========
# ============================================================
def main(cfg_path="configs/train_flux_controlnet.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["logging"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # wandb
    wandb.init(
        project=cfg["logging"].get("wandb_project", "flux-controlnet-training"),
        name=cfg["logging"].get("wandb_run_name", "default"),
        config=cfg,
        resume=cfg["logging"].get("wandb_resume", "allow"),
    )

    # ===== Dataset
    dataset = InpaintPairDataset(
        cfg["data"]["root"],
        size=cfg["data"].get("size", 1024),
        image_suffix=cfg["data"].get("image_suffix", "image_target.png"),
        cond_suffix=cfg["data"].get("cond_suffix", "image_cond.png"),
        mask_suffix=cfg["data"].get("mask_suffix", "mask_cond.png"),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    
    # Prepare validation samples
    val_samples = []
    val_config = cfg.get("validation", {})
    num_val_samples = val_config.get("num_samples", 4)
    if num_val_samples > 0 and len(dataset) > 0:
        val_indices = np.linspace(0, len(dataset)-1, min(num_val_samples, len(dataset)), dtype=int)
        for idx in val_indices:
            val_samples.append(dataset[idx])
        print(f"📊 Prepared {len(val_samples)} validation samples")

    # ===== Base Flux pipeline (for components + scheduler)
    # Use FLUX.1-dev (gated, needs login) or FLUX.1-schnell (open, faster)
    flux_model = cfg.get("model", {}).get("flux_variant", "black-forest-labs/FLUX.1-dev")
    print(f"📦 Loading Flux model: {flux_model}")
    
    pipe = FluxPipeline.from_pretrained(
        flux_model,
        torch_dtype=torch.float32,  # keep fp32 for stability while we set things up
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # ===== CRITICAL: Memory management for 80GB GPU =====
    # Strategy: Keep ONLY what we're training (ControlNet) on GPU
    # Move other models to GPU only when needed
    
    print("📦 Setting up models for memory-efficient training...")
    
    # Move VAE to GPU (needed for encoding latents)
    print("  • VAE → GPU")
    pipe.vae.to(device)
    
    # Keep Transformer on CPU initially (will move to GPU during forward pass)
    print("  • Transformer → CPU (will use on-demand)")
    pipe.transformer.to("cpu")
    
    # Text encoders to GPU temporarily
    print("  • Text Encoders → GPU (temporary)")
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)

    # Freeze all base components
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    
    # ===== Encode text prompts (unconditional - empty prompts)
    # Even for unconditional training, we need proper text embeddings
    print("🔤 Encoding text prompts...")
    with torch.no_grad():
        prompt = ""  # Empty prompt for unconditional training
        prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
    print(f"✅ Text embeddings ready: {prompt_embeds.shape}, pooled: {pooled_prompt_embeds.shape}")
    
    # Offload text encoders to CPU to free up GPU memory (we don't need them anymore)
    print("💾 Offloading text encoders to CPU...")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    del pipe.text_encoder
    del pipe.text_encoder_2
    torch.cuda.empty_cache()
    
    # Check available memory
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"✅ GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    # ===== Initialize FLUX ControlNet (pretrained or from scratch)
    pretrained_controlnet_path = cfg.get("model", {}).get("pretrained_controlnet", None)
    
    if pretrained_controlnet_path:
        print(f"🔧 Loading pretrained FLUX ControlNet from: {pretrained_controlnet_path}")
        try:
            # Load pretrained ControlNet with ignore_mismatched_sizes to handle dimension mismatches
            controlnet = FluxControlNetModel.from_pretrained(
                pretrained_controlnet_path,
                torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True,
                low_cpu_mem_usage=False
            )
            
            # Check if conditioning channels match
            # The input to controlnet_cond_embedding is the control image (e.g., RGB+mask)
            # This goes through controlnet_cond_embedding.conv_in which expects in_channels
            expected_channels = 4  # mask (1) + masked image (3)
            
            # Get actual input channels from the conv_in layer
            if controlnet.input_hint_block is not None:
                actual_channels = controlnet.input_hint_block.conv_in.in_channels
                print(f"📊 ControlNet conditioning embedding input channels: {actual_channels}")
                print(f"📊 ControlNet conditioning embedding output channels: {controlnet.config.conditioning_embedding_channels}")
            else:
                actual_channels = None
                print(f"⚠️ ControlNet has no input_hint_block (conditioning_embedding_channels is None)")
            
            if actual_channels is not None and actual_channels != expected_channels:
                print(f"⚠️ Pretrained ControlNet has {actual_channels} channels, expanding to {expected_channels}...")
                
                # Get the original conv_in layer from conditioning embedding
                old_conv_in = controlnet.input_hint_block.conv_in
                
                # Create new conv layer with correct input channels
                new_conv_in = torch.nn.Conv2d(
                    in_channels=expected_channels,
                    out_channels=old_conv_in.out_channels,
                    kernel_size=old_conv_in.kernel_size,
                    stride=old_conv_in.stride,
                    padding=old_conv_in.padding,
                    bias=old_conv_in.bias is not None
                )
                
                # Initialize new channels
                with torch.no_grad():
                    if actual_channels == 3:
                        # Copy RGB weights, initialize mask channel
                        new_conv_in.weight[:, :3, :, :] = old_conv_in.weight
                        torch.nn.init.xavier_uniform_(new_conv_in.weight[:, 3:, :, :], gain=0.02)
                    else:
                        min_channels = min(actual_channels, expected_channels)
                        new_conv_in.weight[:, :min_channels, :, :] = old_conv_in.weight[:, :min_channels, :, :]
                        if expected_channels > actual_channels:
                            torch.nn.init.xavier_uniform_(new_conv_in.weight[:, min_channels:, :, :], gain=0.02)
                    
                    if new_conv_in.bias is not None and old_conv_in.bias is not None:
                        new_conv_in.bias.copy_(old_conv_in.bias)
                
                # Replace the conv layer
                controlnet.input_hint_block.conv_in = new_conv_in
                print(f"✅ Expanded input layer from {actual_channels} to {expected_channels} channels")
            
            # Inspect the controlnet_cond_embedding structure
            inspect_controlnet_cond_embedding(controlnet)
            
            print(f"✅ Loaded pretrained ControlNet, will finetune it")
            
        except Exception as e:
            print(f"❌ Failed to load pretrained ControlNet: {e}")
            print(f"   Falling back to training from scratch...")
            controlnet = FluxControlNetModel(
                conditioning_embedding_channels=4,
                num_layers=10,
                num_single_layers=20,
                attention_head_dim=64,
                num_attention_heads=12,
            )
    else:
        # CRITICAL: Create a LIGHTWEIGHT ControlNet to fit in 80GB alongside Flux
        print("🎛️ Initializing LIGHTWEIGHT ControlNet from scratch (for 80GB GPU)...")
        controlnet = FluxControlNetModel(
            conditioning_embedding_channels=4,   # RGB + mask
            # Reduced architecture to save memory (~50% smaller)
            num_layers=10,              # Default: 19 → 10
            num_single_layers=20,       # Default: 38 → 20
            attention_head_dim=64,      # Default: 128 → 64
            num_attention_heads=12,     # Default: 24 → 12
        )
    
    print("  • Moving ControlNet to GPU...")
    controlnet.to(device)
    controlnet.to(dtype=torch.bfloat16)
    controlnet.train()
    
    # Enable gradient checkpointing to save memory during training
    controlnet.enable_gradient_checkpointing()
    
    # Check memory after loading ControlNet
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"✅ ControlNet loaded: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    print(f"✅ Gradient checkpointing enabled")

    # Optimizer (only ControlNet)
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
        eps=1e-8,
    )

    # Optional LR schedule
    scheduler = None
    if cfg["train"].get("use_scheduler", False):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg["train"]["epochs"] * len(loader),
            eta_min=cfg["train"].get("min_lr", 1e-7),
        )

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    vae: AutoencoderKL = pipe.vae
    transformer: FluxTransformer2DModel = pipe.transformer

    # All models are in bfloat16 now
    model_dtype = torch.bfloat16

    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        running_loss = 0.0

        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            try:
                images = batch["image"].to(device)  # [B,3,H,W] in [0,1]
                conds = batch["conditioning"].to(device)  # [B,3,H,W]
                masks = batch["mask"].to(device)  # [B,1,H,W]
                batch_size = images.shape[0]

                if torch.isnan(images).any() or torch.isnan(conds).any() or torch.isnan(masks).any():
                    print(f"⚠️ NaNs detected in batch at step {step}, skipping.")
                    continue
                
                # Whiten where mask=1 (same as SDXL)
                conds = torch.where(masks == 1, 1.0, conds)
                controls = torch.cat([conds, masks], dim=1)  # [B, 4, H, W]

                # Expand text embeddings to batch size
                text_embeds_batch = prompt_embeds.repeat(batch_size, 1, 1)
                pooled_embeds_batch = pooled_prompt_embeds.repeat(batch_size, 1)

                # 1) Encode clean images to Flux latents (fp32)
                x0 = encode_latents(images, vae, dtype=torch.float32)   # [B,16,h,w] latents

                # 2) Sample noise & time t for flow-matching
                z = torch.randn_like(x0)                                # Gaussian noise
                t = torch.rand((x0.shape[0],), device=device)           # t ~ U(0,1) - shape [B]

                # Straight path interpolation: x_t = (1 - t) x0 + t z
                # Reshape t for broadcasting: [B] -> [B,1,1,1]
                t_expanded = t.view(-1, 1, 1, 1)
                xt = (1.0 - t_expanded) * x0 + t_expanded * z

                # Target velocity u_t = d/dt x_t = z - x0 (independent of t)
                u_t = z - x0

                # SD3/Flux weighting (helps stability; diffusers util)
                # Use default weighting scheme and pass t as sigmas
                weights = compute_loss_weighting_for_sd3("none", sigmas=t)  # shape [B]
                weights = weights.view(-1, 1, 1, 1)  # [B,1,1,1] for broadcasting

                # Guidance scale for Flux (used in CFG-like mechanism)
                guidance = torch.full((batch_size,), 3.5, device=device, dtype=torch.float32)

                # 3) Run ControlNet to get per-block residuals for the Flux transformer
                #    FluxControlNet expects control image tensors; it internally embeds to sequence.
                ctrl_out = controlnet(
                    hidden_states=xt,                  # latents at time t
                    controlnet_cond=controls,         # our 4-channel control (RGB+mask)
                    timestep=t / 1000.0,              # Flux uses continuous timestep in [0, 0.001]
                    guidance=guidance,                # Guidance scale
                    encoder_hidden_states=text_embeds_batch,
                    pooled_projections=pooled_embeds_batch,
                )

                # The ControlNet returns hidden states added into transformer blocks.
                # We pass them into the Flux transformer.
                # NOTE: in diffusers >=0.32 the kwarg name is `block_controlnet_hidden_states`.
                
                # CRITICAL: Move transformer to GPU for forward pass, then back to CPU
                transformer.to(device)
                pred_v = transformer(
                xt,
                    timestep=t / 1000.0,
                    guidance=guidance,
                encoder_hidden_states=text_embeds_batch,
                pooled_projections=pooled_embeds_batch,
                    block_controlnet_hidden_states=ctrl_out.block_controlnet_hidden_states
                    if hasattr(ctrl_out, "block_controlnet_hidden_states") else None,
                ).sample
                transformer.to("cpu")  # Move back to CPU to save memory
                torch.cuda.empty_cache()

                if not torch.isfinite(pred_v).all():
                    print(f"⚠️ Non-finite pred_v at step {step}, skipping.")
                    continue

                # 4) Flow-matching (velocity) loss with SD3/Flux weighting
                loss = (weights * (pred_v - u_t).pow(2)).mean()

                if not torch.isfinite(loss):
                    print(f"⚠️ NaN loss at step {step}, skipping.")
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Clip ControlNet grads
                grad_norm = torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
                if not math.isfinite(grad_norm):
                    print(f"⚠️ Non-finite grad norm {float(grad_norm):.3f}, skipping step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                running_loss += float(loss.item())
                global_step += 1

                # log
                wandb.log({
                    "train/loss": float(loss.item()),
                    "train/epoch": epoch,
                    "train/step": global_step,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/grad_norm": float(grad_norm),
                }, step=global_step)

                if (step + 1) % cfg["logging"]["print_every"] == 0:
                    print(f"Step {step+1}: loss={loss.item():.4f}, grad_norm={float(grad_norm):.3f}, lr={optimizer.param_groups[0]['lr']:.2e}")

            except Exception as e:
                print(f"❌ Exception at step {step}: {type(e).__name__} - {str(e)}")
                import traceback; traceback.print_exc()
                continue

        avg_loss = running_loss / max(1, len(loader))
        print(f"\n✅ Epoch {epoch} finished. avg_loss={avg_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_loss, "train/epoch_num": epoch}, step=global_step)
        
        # Validation and visualization
        validate_every = val_config.get("validate_every", 5)
        if val_samples and validate_every > 0 and (epoch + 1) % validate_every == 0:
            print(f"\n🎨 Running validation for epoch {epoch+1}...")
            try:
                vis_dir, val_loss = validate_and_visualize_flux(
                    vae=vae,
                    transformer=transformer,
                    controlnet=controlnet,
                    val_samples=val_samples,
                    device=device,
                    epoch=epoch+1,
                    output_dir=cfg["train"]["output_dir"],
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=val_config.get("num_inference_steps", 20),
                    seed=cfg["logging"]["seed"]
                )
                print(f"✅ Validation complete for epoch {epoch+1}")
                print(f"📊 Validation Loss: {val_loss:.4f}")
                
                wandb.log({
                    "validation/loss_epoch": val_loss,
                    "epoch": epoch + 1
                }, step=global_step)
                
            except Exception as e:
                print(f"⚠️ Validation failed: {e}")
                import traceback
                traceback.print_exc()

        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            # Save both PT state_dict and Diffusers format
            ckpt_path = os.path.join(cfg["train"]["output_dir"], f"flux_controlnet_epoch{epoch+1}.pt")
            torch.save(controlnet.state_dict(), ckpt_path)
            print(f"💾 Saved: {ckpt_path}")

            controlnet_dir = os.path.join(cfg["train"]["output_dir"], f"flux_controlnet_epoch{epoch+1}")
            controlnet.save_pretrained(controlnet_dir)
            print(f"💾 Saved diffusers format: {controlnet_dir}")

    # Final save
    final_dir = os.path.join(cfg["train"]["output_dir"], "flux_controlnet_final")
    controlnet.save_pretrained(final_dir)
    torch.save(controlnet.state_dict(), os.path.join(cfg["train"]["output_dir"], "flux_controlnet_final.pt"))
    print("💾 Final model saved!")

    wandb.finish()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Flux ControlNet for 3D Inpainting")
    parser.add_argument("--config", type=str, default="configs/train_flux_controlnet.yaml",
                       help="Path to config YAML file")
    args = parser.parse_args()
    
    main(cfg_path=args.config)
