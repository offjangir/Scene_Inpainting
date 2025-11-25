import os, yaml, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import AutoPipelineForInpainting, DDIMScheduler, ControlNetModel
from diffusers.models.controlnet import ControlNetOutput
from torch.nn import functional as F
from tqdm import tqdm
import random
import numpy as np
import wandb
import math
import os
os.environ["HF_HOME"] = "/scratch/yjangir/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/yjangir/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/yjangir/hf_cache"
os.environ["TORCH_HOME"] = "/scratch/yjangir/torch_cache"

# ============================================================
# === Dataset ================================================
# ============================================================
class InpaintPairDataset(Dataset):
    def __init__(self, root, size=512, image_suffix="_target.png",
                 cond_suffix="_cond.png", mask_suffix="_mask.png"):
        self.root = root
        self.size = size
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
        return {
            "image": img,
            "conditioning_image": cond,
            "mask": mask
        }


# ============================================================
# === Utility ===============================================
# ============================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
        mask_suffix=cfg["data"]["mask_suffix"]
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True
    )

    # Load base pipeline (frozen)
    pipe = AutoPipelineForInpainting.from_pretrained(
        cfg["model"]["pretrained_model"],
        torch_dtype=torch.float32  # Load in fp32 for stability
    ).to(device)

    if cfg["model"]["scheduler"] == "DDIMScheduler":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Freeze the base UNet and VAE (keep in fp32 for numerical stability)
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if hasattr(pipe, 'text_encoder_2'):
        pipe.text_encoder_2.requires_grad_(False)
    
    # Convert UNet to fp16 for inference (VAE stays fp32)
    if cfg["train"]["mixed_precision"] == "fp16":
        pipe.unet.to(dtype=torch.float16)

    # Initialize ControlNet from scratch
    print("🔧 Initializing ControlNet from scratch...")
    controlnet = ControlNetModel.from_unet(
        pipe.unet,
        conditioning_channels=4  # 3 RGB channels + 1 mask channel
    ).to(device)
    
    # Keep ControlNet in fp32 for training stability
    controlnet = controlnet.to(dtype=torch.float32)
    controlnet.train()

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

                    # Encode with fp32 VAE
                    latents = pipe.vae.encode(images * 2 - 1).latent_dist.sample() * 0.18215
                    
                    # ControlNet conditioning: RGB image + mask
                    controlnet_cond = torch.cat([conds, masks], dim=1)  # [B, 4, H, W]

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

                # 4️⃣ Apply ControlNet to UNet
                # Convert to fp16 for UNet inference
                if cfg["train"]["mixed_precision"] == "fp16":
                    noisy_latents_fp16 = noisy_latents.to(dtype=torch.float16)
                    down_block_res_samples = [d.to(dtype=torch.float16) for d in down_block_res_samples]
                    mid_block_res_sample = mid_block_res_sample.to(dtype=torch.float16)
                    added_cond_kwargs_fp16 = {
                        "text_embeds": added_cond_kwargs["text_embeds"].to(dtype=torch.float16),
                        "time_ids": added_cond_kwargs["time_ids"].to(dtype=torch.float16)
                    }
                    encoder_hidden_states_fp16 = torch.zeros(
                        (latents.shape[0], 77, pipe.unet.config.cross_attention_dim),
                        device=device,
                        dtype=torch.float16
                    )
                else:
                    noisy_latents_fp16 = noisy_latents
                    added_cond_kwargs_fp16 = added_cond_kwargs
                    encoder_hidden_states_fp16 = torch.zeros(
                        (latents.shape[0], 77, pipe.unet.config.cross_attention_dim),
                        device=device,
                        dtype=torch.float32
                    )
                
                # UNet forward pass (with gradients from ControlNet residuals)
                noise_pred = pipe.unet(
                    noisy_latents_fp16,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states_fp16,
                    added_cond_kwargs=added_cond_kwargs_fp16,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                # Convert back to fp32 for loss computation
                if cfg["train"]["mixed_precision"] == "fp16":
                    noise_pred = noise_pred.to(dtype=torch.float32)

                # 5️⃣ Compute loss
                if not torch.isfinite(noise_pred).all():
                    print(f"⚠️ NaN/Inf in noise_pred at step {step}, skipping batch.")
                    continue

                loss = F.mse_loss(noise_pred, noise)

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

    wandb.finish()

if __name__ == "__main__":
    main()