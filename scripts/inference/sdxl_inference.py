from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to("cuda")

img_path = "/data/user_data/yjangir/3d_inpainting/data_generated/scene85/pair_0/sample_image_cond.png"
mask_path = "/data/user_data/yjangir/3d_inpainting/data_generated/scene85/pair_0/sample_mask_cond.png"

image = load_image(img_path).resize((1024, 1024))
mask_image = load_image(mask_path).resize((1024, 1024))
 
# invert the mask

prompt = "Robot and kitchen counter scene"
generator = torch.Generator(device="cuda").manual_seed(0)

image = pipe(
  prompt=prompt,
  image=image,
  mask_image=mask_image,
  guidance_scale=8.0,
  num_inference_steps=30,  # steps between 15 and 30 work well for us
  strength=1.0,  # make sure to use `strength` below 1.0
  generator=generator,
).images[0]

image.save("sdxl_inference.png")