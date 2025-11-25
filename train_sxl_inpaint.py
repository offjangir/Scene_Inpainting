"""
ControlNet for Mask-Conditioned Inpainting with Stable Diffusion XL

Training approach:
1. Load SDXL base model (frozen)
2. Initialize ControlNet from scratch with 1-channel input (mask only)
3. Train ControlNet to guide SDXL for inpainting based on binary mask
4. Loss: MSE between predicted and actual noise/velocity in latent space

Architecture:
- Base: Stable Diffusion XL (frozen)
- ControlNet: Trained from scratch, conditioned on mask (1 channel)
- Input: Binary mask (1 = hole to inpaint, 0 = valid region)
- Output: Complete inpainted image

This follows the ControlNet paradigm (like Canny edge) but for inpainting masks.
"""

import os, yaml, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, ControlNetModel
from torch.nn import functional as F
from tqdm import tqdm
import random
import numpy as np
import wandb
import math
import torchvision
import sys
os.environ["HF_HOME"] = "/scratch/yjangir/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/yjangir/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/yjangir/hf_cache"
os.environ["TORCH_HOME"] = "/scratch/yjangir/torch_cache"

# ============================================================
# === Dataset ================================================
# ============================================================
class InpaintPairDataset(Dataset):
    """
    Dataset for ControlNet-based inpainting training.
    
    Training approach:
    - image_target.png: Ground truth image (what we want to generate)
    - image_cond.png: Masked/incomplete image (not used in this approach)
    - mask_cond.png: Binary mask
    
    Mask convention (controlled by invert_mask parameter):
    - If invert_mask=False: 1 = holes to inpaint, 0 = valid regions (default)
    - If invert_mask=True:  0 = holes to inpaint, 1 = valid regions (inverted)
    
    ControlNet learns to guide SDXL to inpaint holes conditioned only on the mask.
    The model sees: mask → ControlNet → guide UNet to generate complete image
    """
    def __init__(self, root, size=512, image_suffix="_target.png",
                 cond_suffix="_cond.png", mask_suffix="_mask.png", invert_mask=False):
        self.root = root
        self.size = size
        self.invert_mask = invert_mask
        self.samples = []
        # Traverse all scene folders
        for scene_dir in sorted(os.listdir(root)):
            scene_path = os.path.join(root, scene_dir)
            if not os.path.isdir(scene_path):
                continue
            
            # Traverse all pair folders inside each scene
            for pair_dir in sorted(os.listdir(scene_path)):
                pair_path = os.path.join(scene_path, pair_dir)
                if not os.path.isdir(pair_path):
                    continue

                img_path  = os.path.join(pair_path, "image_target.png")
                cond_path = os.path.join(pair_path, "image_cond.png")
                mask_path = os.path.join(pair_path, "mask_cond.png")

                # Only add valid samples
                if all(os.path.exists(p) for p in [img_path, cond_path, mask_path]):
                    self.samples.append({
                        "image": img_path,
                        "conditioning": cond_path,  # Loaded but not used in training
                        "mask": mask_path
                    })
                else:
                    print(f"[⚠️] Skipping incomplete pair: {pair_path}")

        print(f"✅ Found {len(self.samples)} valid samples from {root}")

        self.tform_rgb = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        self.tform_mask = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img  = self.tform_rgb(Image.open(s["image"]).convert("RGB"))
        cond = self.tform_rgb(Image.open(s["conditioning"]).convert("RGB"))
        mask = self.tform_mask(Image.open(s["mask"]).convert("L"))
        
        # Invert mask if needed (swap 0s and 1s)
        if self.invert_mask:
            mask = 1.0 - mask
        
        return {
            "image": img,
            "conditioning_image": cond,
            "mask": mask  # Now guaranteed: 1 = hole to inpaint, 0 = valid
        }


# ============================================================
# === Utility ===============================================
# ============================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_and_visualize(
    pipe, controlnet, val_samples, device, epoch, output_dir,
    num_inference_steps=20, guidance_scale=7.5, seed=42
):
    """
    Run validation inference, compute validation loss, and save visualizations
    
    Args:
        pipe: Base SDXL pipeline
        controlnet: Trained ControlNet model
        val_samples: List of validation samples (dicts with 'image', 'mask', 'conditioning_image')
        device: Device to run on
        epoch: Current epoch number
        output_dir: Directory to save visualizations
        num_inference_steps: Number of diffusion steps for inference
        guidance_scale: CFG scale
        seed: Random seed
        
    Returns:
        tuple: (vis_dir, avg_val_loss)
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for server
    import matplotlib.pyplot as plt
    from diffusers import StableDiffusionXLControlNetPipeline
    
    controlnet.eval()
    pipe.unet.eval()  # ✅ FIX: Set UNet to eval mode during validation
    
    # ===== PART 1: Compute Validation Loss =====
    print(f"  📊 Computing validation loss...")
    val_losses = []
    prediction_type = pipe.scheduler.config.prediction_type
    
    with torch.no_grad():
        for idx, sample in enumerate(val_samples):
            try:
                images = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
                masks = sample["mask"].unsqueeze(0).to(device)    # [1, 1, H, W]
                conds = sample["conditioning_image"].unsqueeze(0).to(device)  # [1, 3, H, W]
                
                # Prepare conditioning (same as training)
                conds_val = torch.where(masks == 1, 1, conds)
                
                # Encode to latent space
                latents = pipe.vae.encode(images * 2 - 1).latent_dist.sample() * 0.18215
                controlnet_cond = torch.cat([masks, conds_val], dim=1)  # [1, 4, H, W]
                
                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                # Prepare conditioning for SDXL
                added_cond_kwargs = {
                    "text_embeds": torch.zeros(
                        (latents.shape[0], pipe.text_encoder_2.config.projection_dim),
                        device=device,
                        dtype=torch.float32
                    ),
                    "time_ids": torch.zeros(
                        (latents.shape[0], 6),
                        device=device,
                        dtype=torch.float32
                    )
                }
                
                # ControlNet forward pass
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=torch.zeros(
                        (latents.shape[0], 77, pipe.unet.config.cross_attention_dim),
                        device=device,
                        dtype=torch.float32
                    ),
                    controlnet_cond=controlnet_cond,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )
                
                # UNet forward pass
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=torch.zeros(
                        (latents.shape[0], 77, pipe.unet.config.cross_attention_dim),
                        device=device,
                        dtype=torch.float32
                    ),
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                # Compute loss
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = pipe.scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                
                loss = F.mse_loss(model_pred, target)
                val_losses.append(loss.item())
                
            except Exception as e:
                print(f"  ⚠️ Error computing validation loss for sample {idx}: {e}")
                continue
    
    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
    print(f"  ✅ Validation Loss: {avg_val_loss:.4f} (from {len(val_losses)} samples)")
    
    # ===== PART 2: Generate Visualizations =====
    print(f"  🎨 Generating visualizations...")
    
    # Create inference pipeline
    inference_pipe = StableDiffusionXLControlNetPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        controlnet=controlnet,
        scheduler=pipe.scheduler,
        force_zeros_for_empty_prompt=True
    ).to(device)
    
    # Set to eval mode
    inference_pipe.unet.eval()
    inference_pipe.controlnet.eval()
    
    vis_dir = os.path.join(output_dir, f"epoch_{epoch}_vis")
    os.makedirs(vis_dir, exist_ok=True)
    print(f"  📁 Saving visualizations to: {vis_dir}")
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Process each validation sample
    images_to_log = []
    
    with torch.no_grad():
        for idx, sample in enumerate(val_samples):
            try:
                mask = sample["mask"].unsqueeze(0).to(device)  # [1, 1, H, W]
                gt_image = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
                cond_image = sample["conditioning_image"].unsqueeze(0).to(device)  # [1, 3, H, W]
                
                #  where image is black make it white
                
                cond_image = torch.where(mask == 1, 1, cond_image)
                # Concatenate mask + conditioning image (4 channels total)
                # This gives the model context about what to inpaint
                controlnet_cond = torch.cat([mask, cond_image], dim=1)  # [1, 4, H, W]
                
                # Run inference
                prompt = ""
                negative_prompt = ""
                
                result = inference_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=controlnet_cond,  # Pass mask + masked image (4 channels)
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=mask.shape[2],
                    width=mask.shape[3]
                ).images[0]
                
                # Create visualization grid: Mask | Conditioning | Generated | Ground Truth
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Mask
                axes[0].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
                axes[0].set_title("Input Mask", fontsize=14)
                axes[0].axis('off')
                
                # Conditioning Image (masked image - what model actually sees)
                cond_image = cond_image.squeeze().permute(1, 2, 0).cpu().numpy()
                axes[1].imshow(np.clip(cond_image, 0, 1))
                axes[1].set_title("Input (Masked Image)", fontsize=14)
                axes[1].axis('off')
                
                # Generated
                axes[2].imshow(np.array(result))
                axes[2].set_title("Generated", fontsize=14)
                axes[2].axis('off')
                
                # Ground Truth
                gt_np = gt_image.squeeze().permute(1, 2, 0).cpu().numpy()
                axes[3].imshow(np.clip(gt_np, 0, 1))
                axes[3].set_title("Ground Truth", fontsize=14)
                axes[3].axis('off')
                
                plt.tight_layout()
                
                # Save figure
                save_path = os.path.join(vis_dir, f"sample_{idx}.png")
                save_path = os.path.abspath(save_path)  # Get absolute path
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                # Verify file was saved
                if os.path.exists(save_path):
                    print(f"  ✓ Saved visualization: {save_path}")
                else:
                    print(f"  ❌ Failed to save: {save_path}")
                
                # Prepare for wandb logging
                images_to_log.append(
                    wandb.Image(
                        save_path,
                        caption=f"Epoch {epoch} - Sample {idx}"
                    )
                )
                
            except Exception as e:
                print(f"  ⚠️ Error generating sample {idx}: {e}")
                continue
    
    # Log to wandb
    if images_to_log:
        wandb.log({
            "validation/samples": images_to_log,
            "validation/loss": avg_val_loss,
            "validation/epoch": epoch
        })
        print(f"  📊 Logged {len(images_to_log)} images to W&B")
    
    # Print summary
    saved_files = [f for f in os.listdir(vis_dir) if f.endswith('.png')]
    print(f"  ✅ Validation complete! Saved {len(saved_files)} visualizations to:")
    print(f"     {os.path.abspath(vis_dir)}")
    
    controlnet.train()
    return vis_dir, avg_val_loss


# ============================================================
# === Evaluation (Full Test Set) ============================
# ============================================================
def evaluate_model(pipe, controlnet, test_dataset, device, output_dir, 
                   num_inference_steps=50, guidance_scale=7.5, seed=42):
    """
    Evaluate model on full test set and compute metrics.
    
    Args:
        pipe: Base SDXL pipeline
        controlnet: Trained ControlNet model
        test_dataset: Test dataset
        device: Device to run on
        output_dir: Directory to save results
        num_inference_steps: Number of diffusion steps
        guidance_scale: CFG scale
        seed: Random seed
        
    Returns:
        dict: Evaluation metrics
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from diffusers import StableDiffusionXLControlNetPipeline
    
    print("\n" + "="*60)
    print("🧪 Running Final Evaluation on Test Set")
    print("="*60)
    
    # Set to eval mode
    controlnet.eval()
    pipe.unet.eval()
    
    # Create inference pipeline
    inference_pipe = StableDiffusionXLControlNetPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        controlnet=controlnet,
        scheduler=pipe.scheduler,
        force_zeros_for_empty_prompt=True
    ).to(device)
    
    inference_pipe.unet.eval()
    inference_pipe.controlnet.eval()
    
    # Create output directory
    eval_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Metrics storage
    test_losses = []
    prediction_type = pipe.scheduler.config.prediction_type
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"📊 Evaluating {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Evaluation"):
            try:
                sample = test_dataset[idx]
                
                images = sample["image"].unsqueeze(0).to(device)
                masks = sample["mask"].unsqueeze(0).to(device)
                conds = sample["conditioning_image"].unsqueeze(0).to(device)
                
                # Prepare conditioning
                conds_eval = torch.where(masks == 1, 1, conds)
                
                # ===== Compute Test Loss =====
                latents = pipe.vae.encode(images * 2 - 1).latent_dist.sample() * 0.18215
                controlnet_cond = torch.cat([masks, conds_eval], dim=1)
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                added_cond_kwargs = {
                    "text_embeds": torch.zeros(
                        (latents.shape[0], pipe.text_encoder_2.config.projection_dim),
                        device=device, dtype=torch.float32
                    ),
                    "time_ids": torch.zeros(
                        (latents.shape[0], 6), device=device, dtype=torch.float32
                    )
                }
                
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=torch.zeros(
                        (latents.shape[0], 77, pipe.unet.config.cross_attention_dim),
                        device=device, dtype=torch.float32
                    ),
                    controlnet_cond=controlnet_cond,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )
                
                model_pred = pipe.unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=torch.zeros(
                        (latents.shape[0], 77, pipe.unet.config.cross_attention_dim),
                        device=device, dtype=torch.float32
                    ),
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = pipe.scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                
                loss = F.mse_loss(model_pred, target)
                test_losses.append(loss.item())
                
                # ===== Generate samples for first 20 images =====
                if idx < 20:
                    result = inference_pipe(
                        prompt="",
                        negative_prompt="",
                        image=controlnet_cond,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        height=masks.shape[2],
                        width=masks.shape[3]
                    ).images[0]
                    
                    # Save visualization
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    
                    axes[0].imshow(masks.squeeze().cpu().numpy(), cmap='gray')
                    axes[0].set_title("Mask", fontsize=14)
                    axes[0].axis('off')
                    
                    cond_np = conds_eval.squeeze().permute(1, 2, 0).cpu().numpy()
                    axes[1].imshow(np.clip(cond_np, 0, 1))
                    axes[1].set_title("Input (Masked)", fontsize=14)
                    axes[1].axis('off')
                    
                    axes[2].imshow(np.array(result))
                    axes[2].set_title("Generated", fontsize=14)
                    axes[2].axis('off')
                    
                    gt_np = images.squeeze().permute(1, 2, 0).cpu().numpy()
                    axes[3].imshow(np.clip(gt_np, 0, 1))
                    axes[3].set_title("Ground Truth", fontsize=14)
                    axes[3].axis('off')
                    
                    plt.tight_layout()
                    save_path = os.path.join(eval_dir, f"test_sample_{idx}.png")
                    plt.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.close()
                
            except Exception as e:
                print(f"⚠️ Error evaluating sample {idx}: {e}")
                continue
    
    # Compute metrics
    avg_test_loss = np.mean(test_losses) if test_losses else float('inf')
    std_test_loss = np.std(test_losses) if test_losses else 0.0
    
    metrics = {
        "test_loss_mean": avg_test_loss,
        "test_loss_std": std_test_loss,
        "num_samples": len(test_losses)
    }
    
    # Save metrics
    import json
    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ Evaluation Complete!")
    print("="*60)
    print(f"📊 Test Loss: {avg_test_loss:.4f} ± {std_test_loss:.4f}")
    print(f"📁 Results saved to: {eval_dir}")
    print(f"📄 Metrics saved to: {metrics_path}")
    
    return metrics


# ============================================================
# === Training ===============================================
# ============================================================
def main(cfg_path="configs/train_config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["logging"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize wandb
    wandb.init(
        project=cfg["logging"].get("wandb_project", "controlnet-training"),
        name=cfg["logging"].get("wandb_run_name", "default"),
        config=cfg,
        resume=cfg["logging"].get("wandb_resume", "allow")
    )

    # Dataset & loader
    dataset = InpaintPairDataset(
        cfg["data"]["root"],
        size=cfg["data"]["size"],
        image_suffix=cfg["data"]["image_suffix"],
        cond_suffix=cfg["data"]["cond_suffix"],
        mask_suffix=cfg["data"]["mask_suffix"],
        invert_mask=cfg["data"].get("invert_mask", False)  # Invert mask if True
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True
    )
    
    # Validation config (samples will be randomly selected each epoch)
    val_config = cfg.get("validation", {})
    print(f"📊 Validation: Random sampling enabled ({val_config.get('num_samples', 4)} samples per epoch)")
    
    # Load separate test dataset if specified
    test_dataset = None
    test_root = cfg["data"].get("test_root", None)
    if test_root and test_root != "null" and os.path.exists(test_root):
        print(f"📥 Loading separate TEST dataset from: {test_root}")
        test_dataset = InpaintPairDataset(
            test_root,
            size=cfg["data"]["size"],
            image_suffix=cfg["data"]["image_suffix"],
            cond_suffix=cfg["data"]["cond_suffix"],
            mask_suffix=cfg["data"]["mask_suffix"],
            invert_mask=cfg["data"].get("invert_mask", False)
        )
        print(f"✅ Test dataset loaded: {len(test_dataset)} samples (random sampling per epoch)")
    else:
        print(f"ℹ️ No separate test dataset specified (test_root not found or not configured)")

    # Load base SDXL pipeline (frozen)
    print("📥 Loading Stable Diffusion XL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg["model"]["pretrained_model"],
        torch_dtype=torch.float32,  # Load in fp32 for stability
        variant="fp16" if "fp16" in cfg["model"]["pretrained_model"] else None
    ).to(device)

    if cfg["model"]["scheduler"] == "DDIMScheduler":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Freeze the base UNet and VAE
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if hasattr(pipe, 'text_encoder_2'):
        pipe.text_encoder_2.requires_grad_(False)
    
    # Keep everything in fp32 for stability (no mixed precision issues)
    pipe.unet.to(dtype=torch.float32)
    pipe.vae.to(dtype=torch.float32)

    # Initialize ControlNet from scratch (not loading pretrained)
    print("🔧 Initializing ControlNet from scratch (mask + masked image conditioning)...")
    controlnet = ControlNetModel.from_unet(
        pipe.unet,
        conditioning_channels=4  # 1 mask channel + 3 RGB channels = 4 total
    ).to(device)
    
    # Keep ControlNet in fp32 for training stability
    controlnet = controlnet.to(dtype=torch.float32)
    controlnet.train()
    
    # Check scheduler prediction type
    prediction_type = pipe.scheduler.config.prediction_type
    print(f"📊 Scheduler prediction type: {prediction_type}")

    # Optimizer only for ControlNet
    optimizer = torch.optim.AdamW(
        controlnet.parameters(), 
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
        eps=1e-8  # Prevent division by zero
    )

    # Optional: Learning rate scheduler
    if cfg["train"].get("use_scheduler", False):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=cfg["train"]["epochs"] * len(loader),
            eta_min=cfg["train"].get("min_lr", 1e-7)
        )

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    # Training Loop
    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        running_loss = 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            try:
                images = batch["image"].to(device)
                conds  = batch["conditioning_image"].to(device)
                masks  = batch["mask"].to(device)
                # save images, conds, masks to png
                # Check for NaNs in batch
                if torch.isnan(images).any() or torch.isnan(conds).any() or torch.isnan(masks).any():
                    print(f"⚠️ NaNs detected in batch tensors at step {step}, skipping.")
                    continue

                # 1️⃣ Encode to latent space (VAE in fp32 for stability)
                with torch.no_grad():
                    # Clamp and normalize
                    images = images.clamp(0, 1)
                    conds  = conds.clamp(0, 1)
                    masks  = masks.clamp(0, 1)
                    # save images, conds, masks to png
                    # visualize using torchvision

                    # put conds white where mask is white
                    conds = torch.where(masks == 1, 1, conds)
                    # Encode target images to latents with fp32 VAE
                    latents = pipe.vae.encode(images * 2 - 1).latent_dist.sample() * 0.18215
                    
                    # ControlNet conditioning: mask + masked image (4 channels)
                    # Mask out holes: where mask=1 (white/hole), set image to 0 (black)
                    controlnet_cond = torch.cat([masks, conds], dim=1)  # [B, 4, H, W]
                    # visualize all three images
                # Keep latents in fp32 for ControlNet training stability
                # Only convert for frozen UNet inference

                # Check for abnormal latent magnitudes
                if not torch.isfinite(latents).all():
                    print(f"⚠️ Non-finite latents detected, skipping.")
                    continue
                    
                if latents.abs().max() > 100:
                    print(f"⚠️ Latent explosion detected (max={latents.abs().max().item():.3f}), skipping.")
                    continue

                # 2️⃣ Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # 3️⃣ Get ControlNet output (in fp32)
                # Prepare conditioning for SDXL
                added_cond_kwargs = {
                    "text_embeds": torch.zeros(
                        (latents.shape[0], pipe.text_encoder_2.config.projection_dim),
                        device=device,
                        dtype=torch.float32
                    ),
                    "time_ids": torch.zeros(
                        (latents.shape[0], 6),
                        device=device,
                        dtype=torch.float32
                    )
                }
                
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=torch.zeros(
                        (latents.shape[0], 77, pipe.unet.config.cross_attention_dim),
                        device=device,
                        dtype=torch.float32
                    ),
                    controlnet_cond=controlnet_cond,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )
                
                # Check ControlNet outputs for NaNs
                for i, sample in enumerate(down_block_res_samples):
                    if not torch.isfinite(sample).all():
                        print(f"⚠️ NaN in ControlNet down_block {i}, skipping batch.")
                        raise ValueError("NaN in ControlNet output")
                if not torch.isfinite(mid_block_res_sample).all():
                    print(f"⚠️ NaN in ControlNet mid_block, skipping batch.")
                    raise ValueError("NaN in ControlNet output")

                # 4️⃣ Apply ControlNet to UNet (all in fp32 for stability)
                # UNet forward pass (with gradients from ControlNet residuals)
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=torch.zeros(
                        (latents.shape[0], 77, pipe.unet.config.cross_attention_dim),
                        device=device,
                        dtype=torch.float32
                    ),
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # 5️⃣ Compute loss
                # Determine target based on prediction type
                if prediction_type == "epsilon":
                    # Predict noise
                    target = noise
                elif prediction_type == "v_prediction":
                    # Predict velocity
                    target = pipe.scheduler.get_velocity(latents, noise, timesteps)
                else:
                    # Default to noise prediction
                    target = noise
                
                if not torch.isfinite(model_pred).all():
                    print(f"⚠️ NaN/Inf in model_pred at step {step}, skipping batch.")
                    continue

                loss = F.mse_loss(model_pred, target)

                if not torch.isfinite(loss):
                    print(f"⚠️ NaN loss at step {step}, skipping batch.")
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(controlnet.parameters(), 1.0)
                if not math.isfinite(grad_norm):
                    print(f"⚠️ Non-finite grad norm {grad_norm:.3f}, skipping step.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                
                if cfg["train"].get("use_scheduler", False):
                    scheduler.step()

                running_loss += loss.item()
                global_step += 1

                # ✅ WandB logging
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch": epoch,
                    "train/step": global_step,
                    "train/lr": current_lr,
                    "train/grad_norm": float(grad_norm)
                }, step=global_step)

                if (step + 1) % cfg["logging"]["print_every"] == 0:
                    print(f"Step {step+1}: loss={loss.item():.4f}, grad_norm={grad_norm:.3f}, lr={current_lr:.2e}")

            except ValueError as ve:
                # Catch our custom NaN detection
                continue
            except Exception as e:
                print(f"❌ Exception at step {step}: {type(e).__name__} - {str(e)}")
                import traceback
                traceback.print_exc()
                continue  # Skip bad batch safely

        # End of epoch
        avg_loss = running_loss / max(1, len(loader))
        print(f"\n✅ Epoch {epoch} finished. avg_loss={avg_loss:.4f}")

        wandb.log({
            "train/epoch_loss": avg_loss,
            "train/epoch_num": epoch
        }, step=global_step)

        # Validation and visualization (random samples each epoch for diversity)
        validate_every = val_config.get("validate_every", 5)
        if validate_every > 0 and (epoch + 1) % validate_every == 0:
            # ✅ Randomly select DIFFERENT validation samples from TRAIN set each epoch
            # This provides diverse validation examples across training
            num_val_samples = val_config.get("num_samples", 4)
            val_samples = []
            if num_val_samples > 0 and len(dataset) > 0:
                val_indices = np.random.choice(len(dataset), size=min(num_val_samples, len(dataset)), replace=False)
                for idx in val_indices:
                    val_samples.append(dataset[idx])
                print(f"\n📊 Randomly sampled {len(val_samples)} validation samples from TRAIN set (indices: {val_indices.tolist()})")
            
            if val_samples:
                print(f"\n🎨 Running validation for epoch {epoch+1}...")
                try:
                    # Validate on TRAIN set samples
                    vis_dir, val_loss = validate_and_visualize(
                        pipe=pipe,
                        controlnet=controlnet,
                        val_samples=val_samples,
                        device=device,
                        epoch=epoch+1,
                        output_dir=cfg["train"]["output_dir"],
                        num_inference_steps=val_config.get("num_inference_steps", 20),
                        guidance_scale=val_config.get("guidance_scale", 7.5),
                        seed=cfg["logging"]["seed"]
                    )
                    print(f"✅ Validation (train set) complete for epoch {epoch+1}")
                    print(f"📊 Validation Loss: {val_loss:.4f}")
                    
                    # Log validation loss separately to wandb for easy tracking
                    wandb.log({
                        "validation/loss_epoch": val_loss,
                        "epoch": epoch + 1
                    }, step=global_step)
                    
                except Exception as e:
                    print(f"⚠️ Validation failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ✅ Randomly select DIFFERENT test samples each epoch (diverse unseen examples)
            if test_dataset is not None:
                num_test_samples = val_config.get("num_test_samples", 4)
                test_samples = []
                if num_test_samples > 0 and len(test_dataset) > 0:
                    test_indices = np.random.choice(len(test_dataset), size=min(num_test_samples, len(test_dataset)), replace=False)
                    for idx in test_indices:
                        test_samples.append(test_dataset[idx])
                    print(f"\n📊 Randomly sampled {len(test_samples)} test samples from TEST set (indices: {test_indices.tolist()})")
                
                if test_samples:
                    print(f"\n🧪 Running test evaluation for epoch {epoch+1}...")
                    try:
                        test_vis_dir, test_loss = validate_and_visualize(
                            pipe=pipe,
                            controlnet=controlnet,
                            val_samples=test_samples,
                            device=device,
                            epoch=epoch+1,
                            output_dir=os.path.join(cfg["train"]["output_dir"], "test_eval"),
                            num_inference_steps=val_config.get("num_inference_steps", 20),
                            guidance_scale=val_config.get("guidance_scale", 7.5),
                            seed=cfg["logging"]["seed"]
                        )
                        print(f"✅ Test evaluation complete for epoch {epoch+1}")
                        print(f"📊 Test Loss: {test_loss:.4f}")
                        
                        # Log test loss to wandb
                        wandb.log({
                            "test/loss_epoch": test_loss,
                            "epoch": epoch + 1
                        }, step=global_step)
                        
                    except Exception as e:
                        print(f"⚠️ Test evaluation failed: {e}")
                        import traceback
                        traceback.print_exc()

        # Save checkpoints
        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            ckpt_path = os.path.join(cfg["train"]["output_dir"], f"controlnet_epoch{epoch+1}.pt")
            torch.save(controlnet.state_dict(), ckpt_path)
            print(f"💾 Saved: {ckpt_path}")
            
            # Also save in diffusers format
            controlnet_dir = os.path.join(cfg["train"]["output_dir"], f"controlnet_epoch{epoch+1}")
            controlnet.save_pretrained(controlnet_dir)
            print(f"💾 Saved diffusers format: {controlnet_dir}")

    # Final save
    final_path = os.path.join(cfg["train"]["output_dir"], "controlnet_final.pt")
    torch.save(controlnet.state_dict(), final_path)
    controlnet.save_pretrained(os.path.join(cfg["train"]["output_dir"], "controlnet_final"))
    print(f"💾 Final model saved!")

    # ✅ NEW: Run final evaluation on test set if available
    if test_dataset is not None:
        print("\n" + "="*60)
        print("🧪 Running Final Evaluation on Test Set")
        print("="*60)
        eval_config = cfg.get("evaluation", {})
        try:
            metrics = evaluate_model(
                pipe=pipe,
                controlnet=controlnet,
                test_dataset=test_dataset,
                device=device,
                output_dir=cfg["train"]["output_dir"],
                num_inference_steps=eval_config.get("num_inference_steps", 50),
                guidance_scale=eval_config.get("guidance_scale", 7.5),
                seed=cfg["logging"]["seed"]
            )
            
            # Log to wandb
            wandb.log({
                "test/loss_mean": metrics["test_loss_mean"],
                "test/loss_std": metrics["test_loss_std"],
                "test/num_samples": metrics["num_samples"]
            })
            
        except Exception as e:
            print(f"⚠️ Test evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    wandb.finish()

if __name__ == "__main__":
    main()