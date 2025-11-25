import torch
from diffusers import AutoPipelineForInpainting, DDIMScheduler
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class InpaintPairDataset(Dataset):
    def __init__(self, root):
        self.files = sorted([f.split('_')[0] for f in os.listdir(root) if f.endswith('_target.png')])
        self.root = root
        self.tform = transforms.Compose([
            transforms.ToTensor(),  # converts to [0,1]
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        name = self.files[idx]
        img  = self.tform(Image.open(f"{self.root}/{name}_target.png").convert("RGB"))
        cond = self.tform(Image.open(f"{self.root}/{name}_cond.png").convert("RGB"))
        mask = self.tform(Image.open(f"{self.root}/{name}_mask.png").convert("L"))
        return {"image": img, "conditioning_image": cond, "mask": mask}

dataset = InpaintPairDataset("/data/inpaint_finetune_dataset")
loader  = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = AutoPipelineForInpainting.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()  # optional
pipe.train()

optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-5)

for epoch in range(5):
    for batch in loader:
        images = batch["image"].to("cuda", dtype=torch.float16)
        conds  = batch["conditioning_image"].to("cuda", dtype=torch.float16)
        masks  = batch["mask"].to("cuda", dtype=torch.float16)

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (images.shape[0],), device=images.device)
        noisy_images = pipe.scheduler.add_noise(images, noise, timesteps)

        # Prepare model input
        model_input = torch.cat([noisy_images, masks, conds], dim=1)  # [B, 3+1+3, H, W]
        noise_pred = pipe.unet(model_input, timesteps, encoder_hidden_states=None).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
    torch.save(pipe.unet.state_dict(), f"checkpoints/unet_epoch{epoch}.pt")
