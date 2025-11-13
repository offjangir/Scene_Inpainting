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
    FluxControlNetPipeline,
)
from diffusers.models import FluxControlNetModel, FluxTransformer2DModel
from diffusers.training_utils import compute_loss_weighting_for_sd3
from diffusers.utils.torch_utils import is_compiled_module

import wandb
import os
os.environ["HF_HOME"] = "/scratch/yjangir/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/scratch/yjangir/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/yjangir/hf_cache"
os.environ["TORCH_HOME"] = "/scratch/yjangir/torch_cache"


# ============================================================
# === Dataset (same layout you used) =========================
# ============================================================
class InpaintPairDataset(Dataset):
    def __init__(self, root, size=1024,
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
        img  = self.tform_rgb(Image.open(s["image"]).convert("RGB")).clamp(0,1)
        cond = self.tform_rgb(Image.open(s["conditioning"]).convert("RGB")).clamp(0,1)
        mask = self.tform_mask(Image.open(s["mask"]).convert("L")).clamp(0,1)
        # [B, 4, H, W] control = RGB + mask (match your SDXL setup)
        control = torch.cat([cond, mask], dim=0)
        return {"image": img, "control": control}


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

    # ===== Base Flux pipeline (for components + scheduler)
    pipe = FluxControlNetPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float32,  # keep fp32 for stability while we set things up
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)  # flow-matching scheduler (Flux/SD3)
    pipe.to(device)

    # Freeze base components
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)

    # Optionally put transformer to fp16 for speed; keep ControlNet in fp32
    if cfg["train"]["mixed_precision"] == "fp16":
        pipe.transformer.to(dtype=torch.float16)

    # ===== New Flux ControlNet head from scratch (4-ch control)
    controlnet = FluxControlNetModel(
        conditioning_embedding_channels=4,   # RGB + mask
        # You can also dial these down to make a “lighter” ControlNet:
        # num_layers=19, num_single_layers=38, attention_head_dim=128, num_attention_heads=24, ...
    ).to(device).to(dtype=torch.float32)
    controlnet.train()

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

    # Mixed precision handling for the transformer forward
    use_fp16_transformer = (cfg["train"]["mixed_precision"] == "fp16")

    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        running_loss = 0.0

        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            try:
                images = batch["image"].to(device)        # [B,3,H,W] in [0,1]
                controls = batch["control"].to(device)    # [B,4,H,W] (RGB + mask)

                if torch.isnan(images).any() or torch.isnan(controls).any():
                    print(f"⚠️ NaNs detected in batch at step {step}, skipping.")
                    continue

                # 1) Encode clean images to Flux latents (fp32)
                x0 = encode_latents(images, vae, dtype=torch.float32)   # [B,4,h,w] latents

                # 2) Sample noise & time t for flow-matching
                z = torch.randn_like(x0)                                # Gaussian noise
                t = torch.rand((x0.shape[0], 1, 1, 1), device=device)   # t ~ U(0,1)

                # Straight path interpolation: x_t = (1 - t) x0 + t z
                xt = (1.0 - t) * x0 + t * z

                # Target velocity u_t = d/dt x_t = z - x0 (independent of t)
                u_t = z - x0

                # SD3/Flux weighting (helps stability; diffusers util)
                # compute_loss_weighting_for_sd3 expects "sigmas" in [0,1] like t; use t.squeeze()
                weights = compute_loss_weighting_for_sd3(t.squeeze())  # shape [B,1,1,1] after unsqueeze
                while weights.ndim < u_t.ndim:
                    weights = weights.unsqueeze(-1)

                # 3) Run ControlNet to get per-block residuals for the Flux transformer
                #    FluxControlNet expects control image tensors; it internally embeds to sequence.
                ctrl_out = controlnet(
                    hidden_states=xt,                  # latents at time t
                    controlnet_cond=controls,         # our 4-channel control (RGB+mask)
                    timestep=(t.squeeze() * 1000).long(),  # Flux uses "timestep" for bookkeeping; scale to an int grid
                    encoder_hidden_states=None,       # text conds (we’re training unconditional here)
                )

                # The ControlNet returns hidden states added into transformer blocks.
                # We pass them into the Flux transformer.
                # NOTE: in diffusers >=0.32 the kwarg name is `block_controlnet_hidden_states`.
                transformer_dtype = torch.float16 if use_fp16_transformer else torch.float32
                with torch.autocast(device_type="cuda", dtype=transformer_dtype) if (use_fp16_transformer and torch.cuda.is_available()) else torch.autocast("cpu", enabled=False):
                    pred_v = transformer(
                        xt.to(dtype=transformer_dtype),
                        timestep=(t.squeeze() * 1000).long(),
                        encoder_hidden_states=None,
                        pooled_projections=None,
                        block_controlnet_hidden_states=ctrl_out.block_controlnet_hidden_states
                        if hasattr(ctrl_out, "block_controlnet_hidden_states") else None,
                    ).sample

                # Bring prediction back to fp32 for loss
                pred_v = pred_v.to(torch.float32)

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
    main()
