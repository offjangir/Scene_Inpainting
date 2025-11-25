"""
Fine-tuning Stable Diffusion XL Inpainting Model

Training approach:
1. Load pretrained SDXL inpainting model (diffusers/stable-diffusion-xl-1.0-inpainting-0.1)
2. Fine-tune the UNet on custom 3D scene inpainting data
3. Keep VAE and text encoders frozen (optional: can fine-tune text encoders too)
4. Loss: MSE between predicted and actual noise/velocity in latent space

Architecture:
- Base: Stable Diffusion XL Inpainting (pretrained)
- Fine-tuned: UNet (and optionally text encoders)
- Input: Image + mask (standard inpainting format)
- Output: Inpainted image
"""

import os, yaml, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import AutoPipelineForInpainting, DDIMScheduler
from torch.nn import functional as F
from tqdm import tqdm
import random
import numpy as np
import wandb
import math
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
    Dataset for SDXL inpainting fine-tuning.
    
    Training approach:
    - image_target.png: Ground truth image (what we want to generate)
    - image_cond.png: Masked/incomplete image (RGB with holes)
    - mask_cond.png: Binary mask (1 = hole to inpaint, 0 = valid region)
    
    SDXL inpainting expects:
    - image: The masked/incomplete image
    - mask_image: Binary mask (white = hole, black = valid)
    """
    def __init__(self, root, size=1024, image_suffix="_target.png",
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

                img_path  = os.path.join(pair_path, f"image{image_suffix}")
                cond_path = os.path.join(pair_path, f"image{cond_suffix}")
                mask_path = os.path.join(pair_path, f"mask{mask_suffix}")

                # Only add valid samples
                if all(os.path.exists(p) for p in [img_path, cond_path, mask_path]):
                    self.samples.append({
                        "image": img_path,
                        "conditioning": cond_path,
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
        
        # Invert mask if needed
        if self.invert_mask:
            mask = 1.0 - mask
        
        return {
            "image": img,              # Ground truth
            "conditioning_image": cond,  # Masked image
            "mask": mask               # Binary mask
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
    pipe, val_samples, device, epoch, output_dir,
    num_inference_steps=20, guidance_scale=8.0, seed=42
):
    """
    Run validation inference and save visualizations
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    pipe.unet.eval()
    
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
                
                # Convert to PIL for pipeline
                # Images are in [0, 1] range, convert to [0, 255] for PIL
                cond_pil = transforms.ToPILImage()(cond_image.squeeze(0).cpu().clamp(0, 1))
                # Mask: convert from [0, 1] to [0, 255] for PIL
                mask_for_pil = (mask.squeeze(0).cpu().clamp(0, 1) * 255).byte()
                mask_pil = transforms.ToPILImage()(mask_for_pil)
                
                # Run inference
                result = pipe(
                    prompt="a realistic inpainting",
                    image=cond_pil,
                    mask_image=mask_pil,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
                
                # Create visualization grid
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # Mask
                axes[0].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
                axes[0].set_title("Input Mask", fontsize=14)
                axes[0].axis('off')
                
                # Conditioning Image
                cond_np = cond_image.squeeze().permute(1, 2, 0).cpu().numpy()
                axes[1].imshow(np.clip(cond_np, 0, 1))
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
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                print(f"  ✓ Saved visualization: {save_path}")
                
                # Prepare for wandb
                images_to_log.append(
                    wandb.Image(save_path, caption=f"Epoch {epoch} - Sample {idx}")
                )
                
            except Exception as e:
                print(f"  ⚠️ Error generating sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Log to wandb
    if images_to_log:
        wandb.log({
            "validation/samples": images_to_log,
            "validation/epoch": epoch
        })
        print(f"  📊 Logged {len(images_to_log)} images to W&B")
    
    pipe.unet.train()
    return vis_dir


# ============================================================
# === Training ===============================================
# ============================================================
def main(cfg_path="configs/train_sdxl_inpaint.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg["logging"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize wandb
    wandb.init(
        project=cfg["logging"].get("wandb_project", "sdxl-inpainting-finetune"),
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
        invert_mask=cfg["data"].get("invert_mask", False)
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True
    )
    
    # Prepare validation samples
    val_config = cfg.get("validation", {})
    num_val_samples = val_config.get("num_samples", 4)
    val_samples = []
    if num_val_samples > 0 and len(dataset) > 0:
        val_indices = np.linspace(0, len(dataset)-1, min(num_val_samples, len(dataset)), dtype=int)
        for idx in val_indices:
            val_samples.append(dataset[idx])
        print(f"📊 Prepared {len(val_samples)} validation samples")

    # Load SDXL inpainting pipeline
    print("📥 Loading Stable Diffusion XL Inpainting pipeline...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        cfg["model"]["pretrained_model"],
        torch_dtype=torch.float16 if cfg["train"].get("mixed_precision") == "fp16" else torch.float32,
        variant="fp16" if cfg["train"].get("mixed_precision") == "fp16" else None
    ).to(device)

    if cfg["model"]["scheduler"] == "DDIMScheduler":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Freeze VAE (always frozen)
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    
    # Optionally freeze text encoders
    if cfg["model"].get("freeze_text_encoders", True):
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder.eval()
        if hasattr(pipe, 'text_encoder_2'):
            pipe.text_encoder_2.requires_grad_(False)
            pipe.text_encoder_2.eval()
    
    # UNet is trainable
    pipe.unet.requires_grad_(True)
    pipe.unet.train()
    
    # Set dtypes
    if cfg["train"].get("mixed_precision") == "fp16":
        pipe.unet.to(dtype=torch.float16)
        pipe.vae.to(dtype=torch.float16)
    else:
        pipe.unet.to(dtype=torch.float32)
        pipe.vae.to(dtype=torch.float32)
    
    # Check scheduler prediction type
    prediction_type = pipe.scheduler.config.prediction_type
    print(f"📊 Scheduler prediction type: {prediction_type}")

    # Optimizer for UNet (and optionally text encoders)
    trainable_params = list(pipe.unet.parameters())
    if not cfg["model"].get("freeze_text_encoders", True):
        trainable_params += list(pipe.text_encoder.parameters())
        if hasattr(pipe, 'text_encoder_2'):
            trainable_params += list(pipe.text_encoder_2.parameters())
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.01),
        eps=1e-8
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
    scaler = torch.cuda.amp.GradScaler() if cfg["train"].get("mixed_precision") == "fp16" else None
    
    for epoch in range(cfg["train"]["epochs"]):
        running_loss = 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            try:
                images = batch["image"].to(device)  # Ground truth
                conds  = batch["conditioning_image"].to(device)  # Masked image
                masks  = batch["mask"].to(device)  # Mask

                # Check for NaNs
                if torch.isnan(images).any() or torch.isnan(conds).any() or torch.isnan(masks).any():
                    print(f"⚠️ NaNs detected in batch tensors at step {step}, skipping.")
                    continue

                # 1️⃣ Encode to latent space
                with torch.no_grad():
                    images = images.clamp(0, 1)
                    conds  = conds.clamp(0, 1)
                    masks  = masks.clamp(0, 1)
                    
                    # visualize using torchvision
                    import torchvision
                    torchvision.utils.save_image(images, "images.png")
                    torchvision.utils.save_image(conds, "conds.png")
                    torchvision.utils.save_image(masks, "masks.png")
                    torchvision.utils.save_image(latents, "latents.png")
                    exit()

                    # Encode target images (ground truth) to latents
                    latents = pipe.vae.encode(images * 2 - 1).latent_dist.sample() * 0.18215
                    
                    # Resize mask to latent space dimensions
                    mask_latents = F.interpolate(
                        masks.unsqueeze(1),
                        size=(latents.shape[2], latents.shape[3]),
                        mode="nearest"
                    )  # [B, 1, H_latent, W_latent]

                # Check for abnormal latent magnitudes
                if not torch.isfinite(latents).all():
                    print(f"⚠️ Non-finite latents detected, skipping.")
                    continue
                    
                if latents.abs().max() > 100:
                    print(f"⚠️ Latent explosion detected (max={latents.abs().max().item():.3f}), skipping.")
                    continue

                # 2️⃣ Add noise to ground truth latents
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=device
                ).long()
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # 3️⃣ Prepare text embeddings
                batch_size = latents.shape[0]
                prompt = cfg["train"].get("prompt", "")
                if not prompt:
                    # Empty prompt
                    prompt_embeds, pooled_prompt_embeds = pipe.encode_prompt(
                        prompt=[""] * batch_size,
                        device=device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )
                else:
                    prompt_embeds, pooled_prompt_embeds = pipe.encode_prompt(
                        prompt=[prompt] * batch_size,
                        device=device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                    )

                # 4️⃣ Prepare inpainting inputs for SDXL
                # SDXL inpainting UNet expects 8 channels: 4 latents + 4 mask channels
                # Expand mask from [B, 1, H, W] to [B, 4, H, W] to match latent channels
                mask_latents_expanded = mask_latents.expand(-1, 4, -1, -1)  # [B, 4, H, W]
                
                # Concatenate noisy latents with mask: [B, 4, H, W] + [B, 4, H, W] = [B, 8, H, W]
                noisy_latents_with_mask = torch.cat([noisy_latents, mask_latents_expanded], dim=1)
                
                # Prepare added conditioning kwargs for SDXL
                image_size = cfg["data"]["size"]
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": pipe._get_add_time_ids(
                        (image_size, image_size),
                        (0, 0),
                        (image_size, image_size),
                        dtype=prompt_embeds.dtype,
                    ).to(device)
                }

                # 5️⃣ UNet forward pass
                # SDXL inpainting UNet has 8 input channels (4 latents + 4 mask)
                model_pred = pipe.unet(
                    noisy_latents_with_mask,  # [B, 8, H, W]
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                ).sample

                # 6️⃣ Compute loss
                if prediction_type == "epsilon":
                    target = noise
                elif prediction_type == "v_prediction":
                    target = pipe.scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise
                
                if not torch.isfinite(model_pred).all():
                    print(f"⚠️ NaN/Inf in model_pred at step {step}, skipping batch.")
                    continue

                loss = F.mse_loss(model_pred, target)

                if not torch.isfinite(loss):
                    print(f"⚠️ NaN loss at step {step}, skipping batch.")
                    continue

                optimizer.zero_grad(set_to_none=True)
                
                # Mixed precision training
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    if not math.isfinite(grad_norm):
                        print(f"⚠️ Non-finite grad norm {grad_norm:.3f}, skipping step.")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    if not math.isfinite(grad_norm):
                        print(f"⚠️ Non-finite grad norm {grad_norm:.3f}, skipping step.")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    optimizer.step()
                
                if cfg["train"].get("use_scheduler", False):
                    scheduler.step()

                running_loss += loss.item()
                global_step += 1

                # WandB logging
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

            except Exception as e:
                print(f"❌ Exception at step {step}: {type(e).__name__} - {str(e)}")
                import traceback
                traceback.print_exc()
                continue

        # End of epoch
        avg_loss = running_loss / max(1, len(loader))
        print(f"\n✅ Epoch {epoch} finished. avg_loss={avg_loss:.4f}")

        wandb.log({
            "train/epoch_loss": avg_loss,
            "train/epoch_num": epoch
        }, step=global_step)

        # Validation and visualization
        validate_every = val_config.get("validate_every", 5)
        if validate_every > 0 and (epoch + 1) % validate_every == 0 and val_samples:
            print(f"\n🎨 Running validation for epoch {epoch+1}...")
            try:
                vis_dir = validate_and_visualize(
                    pipe=pipe,
                    val_samples=val_samples,
                    device=device,
                    epoch=epoch+1,
                    output_dir=cfg["train"]["output_dir"],
                    num_inference_steps=val_config.get("num_inference_steps", 20),
                    guidance_scale=val_config.get("guidance_scale", 8.0),
                    seed=cfg["logging"]["seed"]
                )
                print(f"✅ Validation complete for epoch {epoch+1}")
            except Exception as e:
                print(f"⚠️ Validation failed: {e}")
                import traceback
                traceback.print_exc()

        # Save checkpoints
        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            ckpt_path = os.path.join(cfg["train"]["output_dir"], f"unet_epoch{epoch+1}.pt")
            torch.save(pipe.unet.state_dict(), ckpt_path)
            print(f"💾 Saved: {ckpt_path}")
            
            # Also save full pipeline
            pipeline_dir = os.path.join(cfg["train"]["output_dir"], f"pipeline_epoch{epoch+1}")
            pipe.save_pretrained(pipeline_dir)
            print(f"💾 Saved pipeline: {pipeline_dir}")

    # Final save
    final_path = os.path.join(cfg["train"]["output_dir"], "unet_final.pt")
    torch.save(pipe.unet.state_dict(), final_path)
    pipe.save_pretrained(os.path.join(cfg["train"]["output_dir"], "pipeline_final"))
    print(f"💾 Final model saved!")

    wandb.finish()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()

