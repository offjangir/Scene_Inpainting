"""
Inference script for ControlNet Inpainting with Stable Diffusion 1.5

This script loads a trained ControlNet model and performs inpainting inference.
It matches the training setup from train_sd15_inpaint.py.

Usage:
    # Single image inference
    python inference_sd15_inpaint.py \
        --controlnet_path checkpoints/controlnet_final \
        --image_path input.png \
        --mask_path mask.png \
        --output_path output.png

    # Batch inference from directory
    python inference_sd15_inpaint.py \
        --controlnet_path checkpoints/controlnet_final \
        --input_dir data/test \
        --output_dir results

    # With config file
    python inference_sd15_inpaint.py --config configs/inference.yaml
"""

import os
import argparse
import yaml
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    ControlNetModel
)
from tqdm import tqdm
import glob

# Setup cache directories (can be overridden by environment variables)
from utils.env_setup import setup_cache_directories
setup_cache_directories()

# Default checkpoint directory (can be overridden via config or command line)
DEFAULT_CHECKPOINT_DIR = None  # Set via config file or command line argument


def find_latest_checkpoint(checkpoint_dir, use_pt_format=False):
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        use_pt_format: If True, look for .pt files, else look for diffusers format directories
    
    Returns:
        Path to latest checkpoint, or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    if use_pt_format:
        # Look for .pt files
        pattern = os.path.join(checkpoint_dir, "controlnet_epoch*.pt")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            # Try final checkpoint
            final_path = os.path.join(checkpoint_dir, "controlnet_final.pt")
            if os.path.exists(final_path):
                return final_path
            return None
    else:
        # Look for diffusers format directories
        pattern = os.path.join(checkpoint_dir, "controlnet_epoch*")
        checkpoints = [d for d in glob.glob(pattern) if os.path.isdir(d)]
        if not checkpoints:
            # Try final checkpoint
            final_path = os.path.join(checkpoint_dir, "controlnet_final")
            if os.path.exists(final_path):
                return final_path
            return None
    
    # Extract epoch numbers and find latest
    def extract_epoch(path):
        basename = os.path.basename(path)
        if "final" in basename:
            return float('inf')  # Final checkpoint is latest
        try:
            # Extract number from "controlnet_epoch{N}"
            num_str = basename.split("epoch")[1].split(".")[0].split("/")[0]
            return int(num_str)
        except:
            return -1
    
    checkpoints.sort(key=extract_epoch, reverse=True)
    return checkpoints[0] if checkpoints else None


def find_checkpoint_by_epoch(checkpoint_dir, epoch, use_pt_format=False):
    """
    Find a specific checkpoint by epoch number.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        epoch: Epoch number (int) or "final" for final checkpoint
        use_pt_format: If True, look for .pt files, else look for diffusers format
    
    Returns:
        Path to checkpoint, or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    if isinstance(epoch, str) and epoch.lower() == "final":
        if use_pt_format:
            path = os.path.join(checkpoint_dir, "controlnet_final.pt")
        else:
            path = os.path.join(checkpoint_dir, "controlnet_final")
        return path if os.path.exists(path) else None
    
    try:
        epoch_num = int(epoch)
        if use_pt_format:
            path = os.path.join(checkpoint_dir, f"controlnet_epoch{epoch_num}.pt")
        else:
            path = os.path.join(checkpoint_dir, f"controlnet_epoch{epoch_num}")
        return path if os.path.exists(path) else None
    except ValueError:
        return None


def load_controlnet(controlnet_path, base_pipe, device, use_pt_format=False):
    """
    Load ControlNet from either .pt checkpoint or diffusers format.
    
    Args:
        controlnet_path: Path to ControlNet (directory for diffusers, .pt file for checkpoint)
        base_pipe: Base Stable Diffusion pipeline
        device: Device to load on
        use_pt_format: If True, load from .pt file, else load from diffusers format
    
    Returns:
        Loaded ControlNet model
    """
    # Check if path exists
    if not os.path.exists(controlnet_path):
        raise FileNotFoundError(
            f"❌ ControlNet path does not exist: {controlnet_path}\n"
            f"   Please check the path and try again.\n"
            f"   If you have a .pt file, use --use_pt_format flag."
        )
    
    if use_pt_format or controlnet_path.endswith('.pt'):
        print(f"📥 Loading ControlNet from .pt checkpoint: {controlnet_path}")
        if not os.path.isfile(controlnet_path):
            raise FileNotFoundError(f"❌ Expected .pt file but found directory: {controlnet_path}")
        # Initialize ControlNet with same config as training
        controlnet = ControlNetModel.from_unet(
            base_pipe.unet,
            conditioning_channels=3
        ).to(device)
        # Load state dict
        state_dict = torch.load(controlnet_path, map_location=device)
        controlnet.load_state_dict(state_dict)
        print("✅ Loaded ControlNet from .pt checkpoint")
    else:
        print(f"📥 Loading ControlNet from diffusers format: {controlnet_path}")
        if not os.path.isdir(controlnet_path):
            raise FileNotFoundError(
                f"❌ Expected directory for diffusers format but found file: {controlnet_path}\n"
                f"   If this is a .pt file, use --use_pt_format flag."
            )
        # Check if config.json exists (required for diffusers format)
        config_path = os.path.join(controlnet_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"❌ Directory does not contain config.json: {controlnet_path}\n"
                f"   This doesn't appear to be a valid diffusers format checkpoint.\n"
                f"   If you have a .pt file, use --use_pt_format flag."
            )
        try:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_path,
                torch_dtype=torch.float32,
                local_files_only=True  # Only use local files, don't try to download
            ).to(device)
            print(f"✅ Loaded ControlNet with {controlnet.config.conditioning_channels} conditioning channels")
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to load ControlNet from {controlnet_path}\n"
                f"   Error: {str(e)}\n"
                f"   Make sure the directory contains valid diffusers format files.\n"
                f"   If you have a .pt file instead, use --use_pt_format flag."
            ) from e
    
    controlnet.eval()
    return controlnet


def create_conditioning_image(image, mask, invert_mask=False, size=512):
    """
    Create conditioning image by embedding mask into RGB image.
    Matches the training preprocessing.
    
    Args:
        image: PIL Image or tensor [C, H, W] in range [0, 1]
        mask: PIL Image (L mode) or tensor [1, H, W] in range [0, 1]
        invert_mask: If True, invert mask before applying
        size: Target image size (default 512). If None, keeps original size.
    
    Returns:
        Conditioning image tensor [C, H, W] with white pixels where mask=1
    """
    # Convert to tensors if needed
    if isinstance(image, Image.Image):
        if size is None:
            # Keep original size
            tform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            # Resize to specified size
            tform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])
        image = tform(image.convert("RGB"))
    
    if isinstance(mask, Image.Image):
        if size is None:
            # Keep original size
            tform_mask = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            # Resize to specified size
            tform_mask = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
            ])
        mask = tform_mask(mask.convert("L"))
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # [1, H, W]
    
    # Ensure same size
    if image.shape[1:] != mask.shape[1:]:
        resize = transforms.Resize(mask.shape[1:])
        image = resize(image)
    
    # Invert mask if needed
    if invert_mask:
        mask = 1.0 - mask
    
    # Clamp values
    image = image.clamp(0, 1)
    mask = mask.clamp(0, 1)
    
    # Embed mask: white pixels where mask=1
    cond_image = torch.where(mask == 1, torch.ones_like(image), image)
    
    return cond_image


def create_side_by_side_image(input_image, output_image, input_label="Input", output_label="Output"):
    """
    Create a side-by-side comparison image with labels.
    
    Args:
        input_image: PIL Image (input/conditioning image)
        output_image: PIL Image (inpainted result)
        input_label: Label for input image
        output_label: Label for output image
    
    Returns:
        PIL Image with both images side by side
    """
    # Ensure both images have the same height
    h1, w1 = input_image.size[1], input_image.size[0]
    h2, w2 = output_image.size[1], output_image.size[0]
    
    # Resize to same height if needed
    if h1 != h2:
        target_height = max(h1, h2)
        if h1 != target_height:
            input_image = input_image.resize((int(w1 * target_height / h1), target_height), Image.Resampling.LANCZOS)
        if h2 != target_height:
            output_image = output_image.resize((int(w2 * target_height / h2), target_height), Image.Resampling.LANCZOS)
    
    # Get final dimensions
    h = input_image.size[1]
    w1, w2 = input_image.size[0], output_image.size[0]
    total_width = w1 + w2
    
    # Create side-by-side image
    side_by_side = Image.new("RGB", (total_width, h), color="white")
    side_by_side.paste(input_image, (0, 0))
    side_by_side.paste(output_image, (w1, 0))
    
    return side_by_side


def inference_single(
    pipe,
    image_path,
    mask_path,
    output_path,
    num_inference_steps=20,
    guidance_scale=7.5,
    seed=42,
    invert_mask=False,
    size=512,
    keep_original_size=False,
    save_side_by_side=False
):
    """
    Run inference on a single image-mask pair.
    
    Args:
        pipe: StableDiffusionControlNetPipeline
        image_path: Path to input image
        mask_path: Path to mask image
        output_path: Path to save output
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        seed: Random seed
        invert_mask: Whether to invert mask
        size: Image size (will be resized to this, ignored if keep_original_size=True)
        keep_original_size: If True, keeps original image size instead of resizing
        save_side_by_side: If True, also saves input and output side by side
    """
    # Load images
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    
    # Determine size to use
    actual_size = None if keep_original_size else size
    if keep_original_size:
        print(f"📐 Keeping original image size: {image.size}")
    
    # Create conditioning image
    cond_image = create_conditioning_image(image, mask, invert_mask=invert_mask, size=actual_size)
    
    # Convert to PIL for pipeline (pipeline expects PIL or numpy)
    cond_image_pil = transforms.ToPILImage()(cond_image)
    
    # Set generator
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Run inference
    print(f"🎨 Running inference...")
    result = pipe(
        prompt="",  # Empty prompt as in training
        negative_prompt="",
        image=cond_image_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    
    # Save result
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    result.save(output_path)
    print(f"✅ Saved result to: {output_path}")
    
    # Save side-by-side comparison if requested
    if save_side_by_side:
        side_by_side = create_side_by_side_image(cond_image_pil, result)
        # Create side-by-side filename
        base, ext = os.path.splitext(output_path)
        side_by_side_path = f"{base}_comparison{ext}"
        side_by_side.save(side_by_side_path)
        print(f"✅ Saved side-by-side comparison to: {side_by_side_path}")


def inference_batch(
    pipe,
    input_dir,
    output_dir,
    num_inference_steps=20,
    guidance_scale=7.5,
    seed=42,
    invert_mask=False,
    image_suffix="_target.png",
    cond_suffix="_cond.png",
    mask_suffix="_mask.png",
    file_prefix="",  # Prefix for filenames (e.g., "sample_" for "sample_image_cond.png")
    size=512,
    keep_original_size=False,
    save_side_by_side=False
):
    """
    Run batch inference on a directory structure matching the training dataset.
    
    Directory structure expected:
        input_dir/
            scene1/
                pair1/
                    {prefix}image_target.png (or image_suffix)
                    {prefix}image_cond.png (or cond_suffix)
                    {prefix}mask_cond.png (or mask_suffix)
                ...
            ...
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all valid pairs
    samples = []
    for scene_dir in sorted(os.listdir(input_dir)):
        scene_path = os.path.join(input_dir, scene_dir)
        if not os.path.isdir(scene_path):
            continue
        
        for pair_dir in sorted(os.listdir(scene_path)):
            pair_path = os.path.join(scene_path, pair_dir)
            if not os.path.isdir(pair_path):
                continue
            
            # Try to find image and mask files with prefix
            img_path = os.path.join(pair_path, f"{file_prefix}image{image_suffix}")
            cond_path = os.path.join(pair_path, f"{file_prefix}image{cond_suffix}")
            mask_path = os.path.join(pair_path, f"{file_prefix}mask{mask_suffix}")
            
            # Use cond image if available, otherwise use target image
            if os.path.exists(cond_path):
                image_path = cond_path
            elif os.path.exists(img_path):
                image_path = img_path
            else:
                continue
            
            if os.path.exists(mask_path):
                samples.append({
                    "image": image_path,
                    "mask": mask_path,
                    "scene": scene_dir,
                    "pair": pair_dir
                })
    
    print(f"📊 Found {len(samples)} samples to process")
    
    # Process each sample
    for idx, sample in enumerate(tqdm(samples, desc="Processing")):
        try:
            output_subdir = os.path.join(output_dir, sample["scene"], sample["pair"])
            os.makedirs(output_subdir, exist_ok=True)
            output_path = os.path.join(output_subdir, "inpainted.png")
            
            inference_single(
                pipe=pipe,
                image_path=sample["image"],
                mask_path=sample["mask"],
                output_path=output_path,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                invert_mask=invert_mask,
                size=size,
                keep_original_size=keep_original_size,
                save_side_by_side=save_side_by_side
            )
        except Exception as e:
            print(f"⚠️ Error processing {sample['scene']}/{sample['pair']}: {e}")
            continue
    
    print(f"✅ Batch inference complete! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ControlNet Inpainting Inference")
    
    # Model paths
    parser.add_argument("--controlnet_path", type=str, default=None,
                        help="Path to trained ControlNet (diffusers format dir or .pt file). "
                             "If not provided, will use --checkpoint_dir")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help=f"Directory containing checkpoints (default: {DEFAULT_CHECKPOINT_DIR})")
    parser.add_argument("--checkpoint_epoch", type=str, default="latest",
                        help="Epoch number to load (e.g., '100', 'final', or 'latest' for latest checkpoint)")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Base Stable Diffusion model path")
    parser.add_argument("--use_pt_format", action="store_true",
                        help="Load ControlNet from .pt checkpoint instead of diffusers format")
    
    # Input/Output
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to input image (for single inference)")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Path to mask image (for single inference)")
    parser.add_argument("--output_path", type=str, default="output.png",
                        help="Path to save output (for single inference)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory (for batch inference)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory (for batch inference)")
    
    # Inference parameters
    parser.add_argument("--num_inference_steps", type=int, default=20,
                        help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--size", type=int, default=512,
                        help="Image size (ignored if --keep_original_size is set)")
    parser.add_argument("--keep_original_size", action="store_true",
                        help="Keep original image size instead of resizing")
    parser.add_argument("--save_side_by_side", action="store_true",
                        help="Save input and output images side by side for comparison")
    
    # Data parameters
    parser.add_argument("--invert_mask", action="store_true",
                        help="Invert mask (0=holes, 1=valid regions)")
    parser.add_argument("--image_suffix", type=str, default="_target.png",
                        help="Image file suffix for batch inference")
    parser.add_argument("--mask_suffix", type=str, default="_cond.png",
                        help="Mask file suffix for batch inference (default: _cond.png for sample_mask_cond.png)")
    parser.add_argument("--cond_suffix", type=str, default="_cond.png",
                        help="Conditioning image suffix for batch inference")
    parser.add_argument("--file_prefix", type=str, default="sample_",
                        help="Prefix for filenames (e.g., 'sample_' for 'sample_image_cond.png')")
    
    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides command line args)")
    
    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Auto-detected if not specified")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Override args with config values (only if not set via command line)
        for key, value in config.items():
            if hasattr(args, key):
                current_value = getattr(args, key)
                # Use config value if current value is None or default
                if current_value is None or (key == "checkpoint_dir" and current_value == DEFAULT_CHECKPOINT_DIR):
                    setattr(args, key, value)
    
    # Determine device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Resolve checkpoint path
    if args.controlnet_path and args.controlnet_path.lower() != "null":
        controlnet_path = args.controlnet_path
        print(f"📂 Using specified checkpoint: {controlnet_path}")
    else:
        # Use checkpoint directory
        if args.checkpoint_epoch.lower() == "latest":
            controlnet_path = find_latest_checkpoint(args.checkpoint_dir, use_pt_format=args.use_pt_format)
            if controlnet_path:
                print(f"📂 Found latest checkpoint: {controlnet_path}")
            else:
                print(f"❌ No checkpoints found in {args.checkpoint_dir}")
                return
        else:
            controlnet_path = find_checkpoint_by_epoch(
                args.checkpoint_dir, 
                args.checkpoint_epoch, 
                use_pt_format=args.use_pt_format
            )
            if controlnet_path:
                print(f"📂 Found checkpoint for epoch {args.checkpoint_epoch}: {controlnet_path}")
            else:
                print(f"❌ Checkpoint for epoch {args.checkpoint_epoch} not found in {args.checkpoint_dir}")
                return
    
    # Load base SD 1.5 pipeline
    print(f"📥 Loading base Stable Diffusion 1.5 from: {args.base_model}")
    base_pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Use DDIMScheduler (matching training)
    base_pipe.scheduler = DDIMScheduler.from_config(base_pipe.scheduler.config)
    
    # Load ControlNet
    controlnet = load_controlnet(
        controlnet_path,
        base_pipe,
        device,
        use_pt_format=args.use_pt_format
    )
    
    # Create inference pipeline
    print("🔧 Creating ControlNet pipeline...")
    pipe = StableDiffusionControlNetPipeline(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        unet=base_pipe.unet,
        controlnet=controlnet,
        scheduler=base_pipe.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    ).to(device)
    
    pipe.unet.eval()
    pipe.controlnet.eval()
    print("✅ Pipeline ready!")
    
    # Run inference
    if args.input_dir:
        # Batch inference
        print(f"📂 Batch inference mode")
        inference_batch(
            pipe=pipe,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            invert_mask=args.invert_mask,
            image_suffix=args.image_suffix,
            cond_suffix=args.cond_suffix,
            mask_suffix=args.mask_suffix,
            file_prefix=args.file_prefix,
            size=args.size,
            keep_original_size=args.keep_original_size,
            save_side_by_side=args.save_side_by_side
        )
    elif args.image_path and args.mask_path:
        # Single inference
        print(f"🖼️  Single image inference mode")
        inference_single(
            pipe=pipe,
            image_path=args.image_path,
            mask_path=args.mask_path,
            output_path=args.output_path,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            invert_mask=args.invert_mask,
            size=args.size,
            keep_original_size=args.keep_original_size,
            save_side_by_side=args.save_side_by_side
        )
    else:
        print("❌ Error: Either provide --image_path and --mask_path for single inference, "
              "or --input_dir for batch inference")
        return


if __name__ == "__main__":
    main()

