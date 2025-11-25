import os, yaml, torch, random, numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.training_utils import compute_loss_weighting_for_sd3
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

import wandb

os.environ["HF_HOME"] = "/scratch/yjangir/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/yjangir/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/yjangir/hf_cache"
os.environ["TORCH_HOME"] = "/scratch/yjangir/torch_cache"


# ============================================================
# === Latent Packing (CRITICAL FOR FLUX) =====================
# ============================================================
def pack_latents(latents):
    """
    Pack latents from (B, C, H, W) to (B, H//2 * W//2, C*4)
    This is required for FLUX transformer input.
    """
    B, C, H, W = latents.shape
    # Reshape to (B, C, H//2, 2, W//2, 2)
    latents = latents.reshape(B, C, H // 2, 2, W // 2, 2)
    # Permute to (B, H//2, W//2, C, 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    # Reshape to (B, H//2 * W//2, C * 4)
    latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
    return latents


def unpack_latents(latents, h, w):
    """
    Unpack latents from (B, H//2 * W//2, C*4) to (B, C, H, W)
    """
    B, _, C_mul_4 = latents.shape
    C = C_mul_4 // 4
    # Reshape to (B, H//2, W//2, C, 2, 2)
    latents = latents.reshape(B, h // 2, w // 2, C, 2, 2)
    # Permute to (B, C, H//2, 2, W//2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    # Reshape to (B, C, H, W)
    latents = latents.reshape(B, C, h, w)
    return latents


# ============================================================
# === Dataset ================================================
# ============================================================
class InpaintPairDataset(Dataset):
    """
    Dataset where:
    - image_cond.png: Input image with masked area
    - mask_cond.png: Binary mask (1=inpaint, 0=keep)
    - image_target.png: Ground truth complete image
    """
    def __init__(self, root, size=512,
                 image_suffix="image_target.png",
                 cond_suffix="image_cond.png",
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
                target_path = os.path.join(pair_path, image_suffix)
                cond_path = os.path.join(pair_path, cond_suffix)
                mask_path = os.path.join(pair_path, mask_suffix)
                
                if all(os.path.exists(p) for p in [target_path, cond_path, mask_path]):
                    self.samples.append({
                        "target": target_path,
                        "masked_input": cond_path,
                        "mask": mask_path
                    })

        print(f"✅ Found {len(self.samples)} valid samples from {root}")

        self.tform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        
        self.tform_mask = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        target = self.tform(Image.open(s["target"]).convert("RGB")).clamp(0, 1)
        masked_input = self.tform(Image.open(s["masked_input"]).convert("RGB")).clamp(0, 1)
        mask = self.tform_mask(Image.open(s["mask"]).convert("L")).clamp(0, 1)
        
        return {
            "target": target,
            "masked_input": masked_input,
            "mask": mask,
        }


# ============================================================
# === Training ===============================================
# ============================================================
def main(cfg_path="configs/train_flux_controlnet.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg["logging"]["seed"])
    np.random.seed(cfg["logging"]["seed"])
    torch.manual_seed(cfg["logging"]["seed"])
    torch.cuda.manual_seed_all(cfg["logging"]["seed"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project=cfg["logging"].get("wandb_project", "flux-inpaint-training"),
        name=cfg["logging"].get("wandb_run_name", "default"),
        config=cfg,
        resume=cfg["logging"].get("wandb_resume", "allow"),
    )

    # Dataset
    dataset = InpaintPairDataset(
        cfg["data"]["root"],
        size=cfg["data"].get("size", 512),
    )
    
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="vae",
        torch_dtype=torch.bfloat16
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    # Load Transformer
    print("Loading Transformer...")
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    ).to(device)

    # Enable gradient checkpointing
    if cfg["train"].get("gradient_checkpointing", False):
        transformer.enable_gradient_checkpointing()

    # Configure trainable parameters
    transformer.train()
    trainable_params = []
    
    train_all = cfg["train"].get("train_all_transformer", False)
    
    if train_all:
        for param in transformer.parameters():
            param.requires_grad = True
            trainable_params.append(param)
    else:
        # Only train specific layers
        for name, param in transformer.named_parameters():
            if any(x in name for x in ["attn", "norm"]):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
    
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Training {num_trainable:,} parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Scheduler
    scheduler = None
    if cfg["train"].get("use_scheduler", False):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        total_steps = cfg["train"]["epochs"] * len(loader)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=cfg["train"].get("min_lr", 1e-7),
        )

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    # Gradient accumulation
    grad_accum_steps = cfg["train"].get("gradient_accumulation_steps", 1)

    # Training loop
    global_step = 0
    
    for epoch in range(cfg["train"]["epochs"]):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            try:
                target = batch["target"].to(device, dtype=torch.bfloat16)
                masked_input = batch["masked_input"].to(device, dtype=torch.bfloat16)
                mask = batch["mask"].to(device, dtype=torch.bfloat16)

                if torch.isnan(target).any() or torch.isnan(masked_input).any():
                    print(f"⚠️ NaN detected, skipping.")
                    continue

                # Encode to latents
                with torch.no_grad():
                    # Target latents
                    target_latents = vae.encode(target * 2 - 1).latent_dist.sample()
                    target_latents = (target_latents - vae.config.shift_factor) * vae.config.scaling_factor
                    
                    # Masked input latents
                    masked_latents = vae.encode(masked_input * 2 - 1).latent_dist.sample()
                    masked_latents = (masked_latents - vae.config.shift_factor) * vae.config.scaling_factor
                    
                    # Downsample mask to latent resolution
                    latent_h, latent_w = target_latents.shape[-2:]
                    mask_latents = F.interpolate(
                        mask, 
                        size=(latent_h, latent_w),
                        mode='nearest'
                    )

                # Flow-matching: sample noise and time
                noise = torch.randn_like(target_latents)
                bsz = target.shape[0]
                t = torch.rand((bsz, 1, 1, 1), device=device, dtype=torch.bfloat16)
                
                # Straight-path interpolation
                noisy_latents = (1.0 - t) * target_latents + t * noise
                target_velocity = noise - target_latents

                # Apply mask in latent space
                # Where mask=1: use noisy (to be inpainted)
                # Where mask=0: use masked_input (to be preserved)
                inpaint_latents = noisy_latents * mask_latents + masked_latents * (1 - mask_latents)

                # CRITICAL: Pack latents for FLUX
                # (B, 16, H, W) -> (B, H//2 * W//2, 64)
                packed_latents = pack_latents(inpaint_latents)

                # Create img_ids and txt_ids for positional encoding
                # img_ids: (h', w') grid for each token in packed latents
                latent_h_packed = latent_h // 2
                latent_w_packed = latent_w // 2
                
                # Create mesh grid for image tokens
                img_ids = torch.zeros(
                    (bsz, latent_h_packed * latent_w_packed, 3),
                    device=device,
                    dtype=torch.bfloat16
                )
                # Fill with (0, h, w) coordinates
                for h in range(latent_h_packed):
                    for w in range(latent_w_packed):
                        idx = h * latent_w_packed + w
                        img_ids[:, idx, 0] = 0  # t (always 0 for images)
                        img_ids[:, idx, 1] = h  # h coordinate
                        img_ids[:, idx, 2] = w  # w coordinate
                
                # txt_ids: all zeros (0, 0, 0) for text tokens
                max_txt_tokens = 512  # T5 max tokens
                txt_ids = torch.zeros(
                    (bsz, max_txt_tokens, 3),
                    device=device,
                    dtype=torch.bfloat16
                )

                # Empty text embeddings (unconditional)
                prompt_embeds = torch.zeros(
                    (bsz, max_txt_tokens, 4096),
                    device=device,
                    dtype=torch.bfloat16
                )
                pooled_prompt_embeds = torch.zeros(
                    (bsz, 768),
                    device=device,
                    dtype=torch.bfloat16
                )

                # Guidance scale (use 1.0 for unconditional training or 3.5 for CFG)
                guidance_vec = torch.full(
                    (bsz,),
                    1.0,  # No guidance during training
                    device=device,
                    dtype=torch.bfloat16
                )

                # Convert t to timesteps (FLUX uses 0-1 range, not 0-1000)
                timesteps = t.squeeze()  # Keep in [0, 1] range

                # Forward through transformer
                # Pass all required arguments in the correct order
                pred_velocity = transformer(
                    hidden_states=packed_latents,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=timesteps,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance_vec,
                    return_dict=True,
                ).sample

                # Unpack predicted velocity
                pred_velocity = unpack_latents(pred_velocity, latent_h, latent_w)

                # Flow-matching loss with SD3 weighting
                weights = compute_loss_weighting_for_sd3(t.squeeze())
                while weights.ndim < pred_velocity.ndim:
                    weights = weights.unsqueeze(-1)
                
                loss = (weights * (pred_velocity.float() - target_velocity.float()).pow(2)).mean()
                loss = loss / grad_accum_steps

                if not torch.isfinite(loss):
                    print(f"⚠️ Non-finite loss, skipping.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss.backward()

                # Gradient accumulation
                if (step + 1) % grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    
                    if not torch.isfinite(grad_norm):
                        print(f"⚠️ Non-finite grad norm, skipping.")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    if scheduler:
                        scheduler.step()
                    
                    global_step += 1

                    wandb.log({
                        "train/loss": loss.item() * grad_accum_steps,
                        "train/grad_norm": float(grad_norm),
                        "train/lr": optimizer.param_groups[0]['lr'],
                        "train/epoch": epoch,
                    }, step=global_step)

                running_loss += loss.item() * grad_accum_steps

                if (step + 1) % cfg["logging"]["print_every"] == 0:
                    print(f"Step {step+1}/{len(loader)}: "
                          f"loss={loss.item() * grad_accum_steps:.4f}, "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}")

            except Exception as e:
                print(f"❌ Exception at step {step}: {type(e).__name__} - {str(e)}")
                import traceback
                traceback.print_exc()
                optimizer.zero_grad(set_to_none=True)
                continue

        avg_loss = running_loss / max(1, len(loader))
        print(f"\n✅ Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        wandb.log({
            "train/epoch_loss": avg_loss,
            "train/epoch_num": epoch
        }, step=global_step)

        # Save checkpoint
        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            save_dir = os.path.join(
                cfg["train"]["output_dir"],
                f"flux_transformer_epoch{epoch+1}"
            )
            transformer.save_pretrained(save_dir)
            print(f"💾 Checkpoint saved: {save_dir}")

    # Final save
    final_dir = os.path.join(cfg["train"]["output_dir"], "flux_transformer_final")
    transformer.save_pretrained(final_dir)
    print("✅ Training complete! Final model saved.")

    wandb.finish()


if __name__ == "__main__":
    main()