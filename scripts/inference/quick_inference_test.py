"""Quick inference test on a single sample"""
import os
import sys

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, DDIMScheduler
import matplotlib.pyplot as plt

# Setup cache directories
from utils.env_setup import setup_cache_directories
setup_cache_directories()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths (update these for your setup)
# TODO: Make these configurable via command line arguments or config file
controlnet_path = "/data/group_data/katefgroup/datasets/yjangir/3d_inpaint/checkpoints_long/controlnet_epoch176.pt"
mask_path = "data_generated/scene85/pair_0/sample_mask_cond.png"
cond_path = "data_generated/scene85/pair_0/sample_image_cond.png"
target_path = "data_generated/scene85/pair_0/sample_image_target.png"

print("\n" + "="*70)
print("Loading ControlNet...")
print("="*70)

# Load trained ControlNet
from diffusers import StableDiffusionXLPipeline

temp_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float32
)
controlnet = ControlNetModel.from_unet(
    temp_pipe.unet,
    conditioning_channels=4
)

# Load state dict
print(f"Loading checkpoint: {controlnet_path}")
state_dict = torch.load(controlnet_path, map_location="cpu")
controlnet.load_state_dict(state_dict)
print("✓ ControlNet loaded")

# Create inference pipeline
print("Creating pipeline...")
pipe = StableDiffusionXLControlNetPipeline(
    vae=temp_pipe.vae,
    text_encoder=temp_pipe.text_encoder,
    text_encoder_2=temp_pipe.text_encoder_2,
    tokenizer=temp_pipe.tokenizer,
    tokenizer_2=temp_pipe.tokenizer_2,
    unet=temp_pipe.unet,
    controlnet=controlnet,
    scheduler=temp_pipe.scheduler,
    force_zeros_for_empty_prompt=True
)
pipe = pipe.to(device)
pipe.unet.eval()
pipe.controlnet.eval()

print("✓ Pipeline ready")
del temp_pipe

print("\n" + "="*70)
print("Loading inputs...")
print("="*70)

# Load images
mask_img = Image.open(mask_path).convert("L")
cond_img = Image.open(cond_path).convert("RGB")
target_img = Image.open(target_path).convert("RGB")

# Get size
# size = 640
# original_size = mask_img.size
# print(f"Original size: {original_size}, Processing size: {size}x{size}")

# # Transform
# transform = transforms.Compose([
#     transforms.Resize((size, size)),
#     transforms.ToTensor(),
# ])

# mask = transform(mask_img).unsqueeze(0).to(device)  # [1, 1, H, W]
# cond = transform(cond_img).unsqueeze(0).to(device)  # [1, 3, H, W]

size = 512
# Transform
transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
])

mask = transform(mask_img).unsqueeze(0).to(device)  # [1, 1, H, W]
cond = transform(cond_img).unsqueeze(0).to(device)  # [1, 3, H, W]

print(f"Mask shape: {mask.shape}")
print(f"Cond shape: {cond.shape}")

print("\n" + "="*70)
print("Running inference...")
print("="*70)

# Prepare conditioning (same as training)
cond_processed = torch.where(mask == 1, torch.ones_like(cond), cond)
controlnet_cond = torch.cat([mask, cond_processed], dim=1)  # [1, 4, H, W]

print(f"ControlNet input shape: {controlnet_cond.shape}")

# Generate
with torch.no_grad():
    result = pipe(
        prompt="",
        negative_prompt="",
        image=controlnet_cond,
        num_inference_steps=88,
        guidance_scale=7.5,
        generator=torch.Generator(device=device).manual_seed(42),
        height=size,
        width=size
    ).images[0]

print("✓ Generation complete!")

# Resize to original if needed
# if original_size != (size, size):
#     result = result.resize(original_size, Image.LANCZOS)

print("\n" + "="*70)
print("Saving results...")
print("="*70)

# Save visualization
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Mask
axes[0].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
axes[0].set_title("Input Mask", fontsize=14)
axes[0].axis('off')

# Conditioning
cond_np = cond_processed.squeeze().permute(1, 2, 0).cpu().numpy()
axes[1].imshow(np.clip(cond_np, 0, 1))
axes[1].set_title("Input (Masked)", fontsize=14)
axes[1].axis('off')

# Generated
axes[2].imshow(result)
axes[2].set_title("Generated", fontsize=14)
axes[2].axis('off')

# Ground Truth
axes[3].imshow(target_img)
axes[3].set_title("Ground Truth", fontsize=14)
axes[3].axis('off')

plt.tight_layout()
os.makedirs("test_inference_output", exist_ok=True)
output_path = "test_inference_output/test_result.png"
plt.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close()

print(f"✓ Saved visualization to: {output_path}")

# Also save just the generated image
result.save("test_inference_output/generated.png")
print(f"✓ Saved generated image to: test_inference_output/generated.png")

print("\n" + "="*70)
print("✅ Done!")
print("="*70)

