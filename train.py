import os, yaml, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import AutoPipelineForInpainting, DDIMScheduler
from torch.nn import functional as F
from tqdm import tqdm
import random
import numpy as np

# ============================================================
# === Dataset ================================================
# ============================================================
class InpaintPairDataset(Dataset):
    def __init__(self, root, size=512, image_suffix="_target.png",
                 cond_suffix="_cond.png", mask_suffix="_mask.png"):
        self.root = root
        self.size = size
        self.image_suffix = image_suffix
        self.cond_suffix  = cond_suffix
        self.mask_suffix  = mask_suffix
        self.files = sorted([
            f.split(image_suffix)[0]
            for f in os.listdir(root)
            if f.endswith(image_suffix)
        ])
        self.tform_rgb = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        self.tform_mask = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img  = self.tform_rgb(Image.open(f"{self.root}/{name}{self.image_suffix}").convert("RGB"))
        cond = self.tform_rgb(Image.open(f"{self.root}/{name}{self.cond_suffix}").convert("RGB"))
        mask = self.tform_mask(Image.open(f"{self.root}/{name}{self.mask_suffix}").convert("L"))
        return {
            "image": img,                 # ground truth
            "conditioning_image": cond,   # masked input
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

    # Model
    pipe = AutoPipelineForInpainting.from_pretrained(
        cfg["model"]["pretrained_model"],
        torch_dtype=torch.float16 if cfg["train"]["mixed_precision"] == "fp16" else torch.float32
    ).to(device)

    if cfg["model"]["scheduler"] == "DDIMScheduler":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if cfg["model"]["cpu_offload"]:
        pipe.enable_model_cpu_offload()

    pipe.train()
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=cfg["train"]["lr"])

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    # Training Loop
    for epoch in range(cfg["train"]["epochs"]):
        running_loss = 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            images = batch["image"].to(device, dtype=torch.float16)
            conds  = batch["conditioning_image"].to(device, dtype=torch.float16)
            masks  = batch["mask"].to(device, dtype=torch.float16)

            # Forward diffusion
            noise = torch.randn_like(images)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps,
                (images.shape[0],), device=device
            ).long()
            noisy_images = pipe.scheduler.add_noise(images, noise, timesteps)

            # UNet input: [B, (3+1+3), H, W]
            model_input = torch.cat([noisy_images, masks, conds], dim=1)

            # Predict noise
            noise_pred = pipe.unet(
                model_input,
                timesteps,
                encoder_hidden_states=None
            ).sample

            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % cfg["logging"]["print_every"] == 0:
                print(f"Step {step+1}: loss={loss.item():.4f}")

        avg_loss = running_loss / len(loader)
        print(f"\n✅ Epoch {epoch} finished. avg_loss={avg_loss:.4f}")

        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            ckpt_path = os.path.join(cfg["train"]["output_dir"], f"unet_epoch{epoch+1}.pt")
            torch.save(pipe.unet.state_dict(), ckpt_path)
            print(f"💾 Saved: {ckpt_path}")

if __name__ == "__main__":
    main()
