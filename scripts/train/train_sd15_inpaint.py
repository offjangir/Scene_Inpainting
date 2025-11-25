"""
ControlNet Inpainting with Stable Diffusion 1.5

Uses pretrained inpainting ControlNet (lllyasviel/control_v11p_sd15_inpaint)
Fine-tuned on custom 3D scene inpainting data

Training approach:
1. Load SD 1.5 base model (frozen)
2. Load pretrained inpainting ControlNet
3. Condition on masked RGB image (holes marked white/black)
4. Fine-tune ControlNet on your custom data
5. Loss: MSE between predicted and actual noise in latent space

Architecture:
- Base: Stable Diffusion 1.5 (frozen)
- ControlNet: Pretrained inpainting ControlNet, fine-tuned
- Input: Masked RGB image (3 channels)
- Output: Complete inpainted image
"""

import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, DDIMScheduler, ControlNetModel
from torch.nn import functional as F
from tqdm import tqdm
import random
import numpy as np
import wandb
import math
import re
import glob

# Setup cache directories (can be overridden by environment variables)
from utils.env_setup import setup_cache_directories
setup_cache_directories()

# ============================================================
# === Dataset ================================================
# ============================================================
class InpaintPairDataset(Dataset):
    """
    Dataset for ControlNet-based inpainting training.
    
    Training approach:
    - image_target.png: Ground truth image (what we want to generate)
    - image_cond.png: Masked/incomplete image (RGB with holes masked out)
    - mask_cond.png: Binary mask (used to create conditioning image)
    
    Mask convention (controlled by invert_mask parameter):
    - If invert_mask=False: 1 = holes to inpaint, 0 = valid regions (default)
    - If invert_mask=True:  0 = holes to inpaint, 1 = valid regions (inverted)
    
    ControlNet learns to guide SD to inpaint holes conditioned on masked RGB image.
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


def find_last_checkpoint(output_dir):
    """
    Find the last checkpoint in the output directory.
    Looks for files matching pattern: controlnet_epoch{epoch}.pt
    
    Returns:
        tuple: (checkpoint_path, epoch_number) or (None, 0) if not found
    """
    if not os.path.exists(output_dir):
        return None, 0
    
    # Find all checkpoint files
    pattern = os.path.join(output_dir, "controlnet_epoch*.pt")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        return None, 0
    
    # Extract epoch numbers and find the latest
    epoch_numbers = []
    for ckpt_file in checkpoint_files:
        match = re.search(r'controlnet_epoch(\d+)\.pt', ckpt_file)
        if match:
            epoch_numbers.append((int(match.group(1)), ckpt_file))
    
    if not epoch_numbers:
        return None, 0
    
    # Sort by epoch number and return the latest
    epoch_numbers.sort(key=lambda x: x[0], reverse=True)
    latest_epoch, latest_ckpt = epoch_numbers[0]
    
    return latest_ckpt, latest_epoch


def load_checkpoint(checkpoint_path, controlnet, optimizer=None, scheduler=None, device="cuda"):
    """
    Load checkpoint including model state, optimizer state, and training state.
    
    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        controlnet: ControlNet model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load checkpoint on
    
    Returns:
        dict: Training state (epoch, global_step) or None if checkpoint doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"📥 Loading checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        if isinstance(checkpoint, dict):
            # Full checkpoint with training state
            if "model_state_dict" in checkpoint:
                controlnet.load_state_dict(checkpoint["model_state_dict"])
                print("✅ Loaded model state")
            elif "state_dict" in checkpoint:
                controlnet.load_state_dict(checkpoint["state_dict"])
                print("✅ Loaded model state")
            else:
                # Assume it's just the model state dict
                controlnet.load_state_dict(checkpoint)
                print("✅ Loaded model state (state dict only)")
            
            # Load optimizer state if available
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("✅ Loaded optimizer state")
            
            # Load scheduler state if available
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("✅ Loaded scheduler state")
            
            # Extract training state
            training_state = {
                "epoch": checkpoint.get("epoch", 0),
                "global_step": checkpoint.get("global_step", 0),
            }
            
            if training_state["epoch"] > 0 or training_state["global_step"] > 0:
                print(f"📊 Resuming from epoch {training_state['epoch']}, step {training_state['global_step']}")
            
            return training_state
        else:
            # Just model state dict
            controlnet.load_state_dict(checkpoint)
            print("✅ Loaded model state (state dict only)")
            return {"epoch": 0, "global_step": 0}
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def validate_and_visualize(
    pipe, controlnet, val_samples, device, epoch, output_dir,
    num_inference_steps=20, guidance_scale=7.5, seed=42
):
    """
    Run validation inference and save visualizations
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    controlnet.eval()
    pipe.unet.eval()
    
    # Create inference pipeline
    inference_pipe = StableDiffusionControlNetPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
        unet=pipe.unet,
        controlnet=controlnet,
        scheduler=pipe.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    ).to(device)
    
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
                
                # Make masked regions white (embed mask info in RGB)
                cond_image = torch.where(mask == 1, 1, cond_image)
                
                # Run inference
                result = inference_pipe(
                    prompt="",
                    negative_prompt="",
                    image=cond_image,  # RGB masked image (3 channels)
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
                continue
    
    # Log to wandb
    if images_to_log:
        wandb.log({
            "validation/samples": images_to_log,
            "validation/epoch": epoch
        })
        print(f"  📊 Logged {len(images_to_log)} images to W&B")
    
    controlnet.train()
    return vis_dir


# ============================================================
# === Training ===============================================
# ============================================================
def main(cfg_path="configs/train_sd15_inpaint.yaml"):
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

    # Load base SD 1.5 pipeline (frozen)
    print("📥 Loading Stable Diffusion 1.5 pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg["model"]["pretrained_model"],
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)

    if cfg["model"]["scheduler"] == "DDIMScheduler":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Freeze the base UNet, VAE, and text encoder
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    
    pipe.unet.to(dtype=torch.float32)
    pipe.vae.to(dtype=torch.float32)

    # Load pretrained inpainting ControlNet
    pretrained_controlnet_path = cfg["model"].get("pretrained_controlnet", None)
    
    if pretrained_controlnet_path and pretrained_controlnet_path != "null":
        print(f"🔧 Loading pretrained inpainting ControlNet from: {pretrained_controlnet_path}")
        print("   This ControlNet is specifically trained for inpainting!")
        try:
            controlnet = ControlNetModel.from_pretrained(
                pretrained_controlnet_path,
                torch_dtype=torch.float32
            ).to(device)
            print(f"✅ Loaded pretrained inpainting ControlNet with {controlnet.config.conditioning_channels} channels")
        except Exception as e:
            print(f"❌ Failed to load pretrained ControlNet: {e}")
            print("🔧 Falling back to training from scratch...")
            controlnet = ControlNetModel.from_unet(
                pipe.unet,
                conditioning_channels=3
            ).to(device)
    else:
        print("🔧 Initializing ControlNet from scratch (RGB conditioning)...")
        controlnet = ControlNetModel.from_unet(
            pipe.unet,
            conditioning_channels=3
        ).to(device)
    
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
        eps=1e-8
    )

    # Optional: Learning rate scheduler
    scheduler = None
    if cfg["train"].get("use_scheduler", False):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=cfg["train"]["epochs"] * len(loader),
            eta_min=cfg["train"].get("min_lr", 1e-7)
        )

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    # Checkpoint loading
    start_epoch = 0
    global_step = 0
    resume_from_checkpoint = cfg["train"].get("resume_from_checkpoint", None)
    
    if resume_from_checkpoint:
        # Explicit checkpoint path provided
        if resume_from_checkpoint.lower() == "auto" or resume_from_checkpoint.lower() == "last":
            # Auto-detect last checkpoint
            checkpoint_path, epoch_num = find_last_checkpoint(cfg["train"]["output_dir"])
            if checkpoint_path:
                resume_from_checkpoint = checkpoint_path
            else:
                print("⚠️ No checkpoint found for auto-resume, starting from scratch")
                resume_from_checkpoint = None
        else:
            # Use provided path
            if not os.path.exists(resume_from_checkpoint):
                print(f"⚠️ Checkpoint path not found: {resume_from_checkpoint}, starting from scratch")
                resume_from_checkpoint = None
    
    if resume_from_checkpoint:
        training_state = load_checkpoint(
            resume_from_checkpoint, 
            controlnet, 
            optimizer=optimizer, 
            scheduler=scheduler,
            device=device
        )
        if training_state:
            start_epoch = training_state["epoch"]
            global_step = training_state["global_step"]
            print(f"🔄 Resuming training from epoch {start_epoch}, global_step {global_step}")
        else:
            print("⚠️ Failed to load checkpoint, starting from scratch")
    else:
        print("🆕 Starting training from scratch")

    # Training Loop
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
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

                # 1️⃣ Encode to latent space
                with torch.no_grad():
                    # Clamp and normalize
                    images = images.clamp(0, 1)
                    conds  = conds.clamp(0, 1)
                    masks  = masks.clamp(0, 1)

                    # Put conds white where mask is white (embed mask in RGB)
                    conds = torch.where(masks == 1, 1, conds)
                    
                    # Encode target images to latents
                    latents = pipe.vae.encode(images * 2 - 1).latent_dist.sample() * 0.18215
                    
                    # ControlNet conditioning: RGB masked image only (3 channels)
                    controlnet_cond = conds  # [B, 3, H, W]

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

                # 3️⃣ Get ControlNet output
                # Get text embeddings (empty prompt)
                # Tokenize empty string properly for batch
                batch_size = latents.shape[0]
                text_inputs = pipe.tokenizer(
                    [""] * batch_size,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                encoder_hidden_states = pipe.text_encoder(
                    text_inputs.input_ids.to(device)
                )[0]
                
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
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
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # 5️⃣ Compute loss
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

            except ValueError as ve:
                continue
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
                    controlnet=controlnet,
                    val_samples=val_samples,
                    device=device,
                    epoch=epoch+1,
                    output_dir=cfg["train"]["output_dir"],
                    num_inference_steps=val_config.get("num_inference_steps", 20),
                    guidance_scale=val_config.get("guidance_scale", 7.5),
                    seed=cfg["logging"]["seed"]
                )
                print(f"✅ Validation complete for epoch {epoch+1}")
            except Exception as e:
                print(f"⚠️ Validation failed: {e}")
                import traceback
                traceback.print_exc()

        # Save checkpoints
        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            ckpt_path = os.path.join(cfg["train"]["output_dir"], f"controlnet_epoch{epoch+1}.pt")
            
            # Save full checkpoint with training state
            checkpoint_dict = {
                "model_state_dict": controlnet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
            }
            if scheduler is not None:
                checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()
            
            torch.save(checkpoint_dict, ckpt_path)
            print(f"💾 Saved: {ckpt_path}")
            
            # Also save in diffusers format
            controlnet_dir = os.path.join(cfg["train"]["output_dir"], f"controlnet_epoch{epoch+1}")
            controlnet.save_pretrained(controlnet_dir)
            print(f"💾 Saved diffusers format: {controlnet_dir}")

    # Final save
    final_path = os.path.join(cfg["train"]["output_dir"], "controlnet_final.pt")
    final_checkpoint_dict = {
        "model_state_dict": controlnet.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": cfg["train"]["epochs"],
        "global_step": global_step,
    }
    if scheduler is not None:
        final_checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(final_checkpoint_dict, final_path)
    controlnet.save_pretrained(os.path.join(cfg["train"]["output_dir"], "controlnet_final"))
    print(f"💾 Final model saved!")

    wandb.finish()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()

