"""
Inference Script for ControlNet-based Inpainting with Stable Diffusion XL

This script performs inpainting using a trained ControlNet model conditioned on masks.

Usage Examples:
    
    # Single image inference
    python inference.py \
        --controlnet_path output/controlnet_final \
        --mask_path path/to/mask.png \
        --cond_path path/to/conditioning.png \
        --output_path output/result.png
    
    # Batch inference on directory
    python inference.py \
        --controlnet_path output/controlnet_final \
        --input_dir data/test \
        --output_dir output/results \
        --batch
    
    # With custom parameters
    python inference.py \
        --controlnet_path output/controlnet_final \
        --mask_path mask.png \
        --cond_path cond.png \
        --output_path result.png \
        --num_steps 50 \
        --guidance_scale 7.5 \
        --seed 42
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, DDIMScheduler
from tqdm import tqdm
import warnings

# Setup cache directories (can be overridden by environment variables)
from utils.env_setup import setup_cache_directories
setup_cache_directories()

warnings.filterwarnings('ignore')


class InpaintingInference:
    """
    Inference class for ControlNet-based inpainting.
    """
    
    def __init__(
        self,
        controlnet_path,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda",
        use_ddim=True,
        torch_dtype=torch.float32
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            controlnet_path: Path to trained ControlNet (diffusers format or .pt file)
            base_model: Base SDXL model identifier or path
            device: Device to run on ('cuda' or 'cpu')
            use_ddim: Whether to use DDIM scheduler
            torch_dtype: Data type for inference (torch.float32 or torch.float16)
        """
        self.device = device
        self.torch_dtype = torch_dtype
        
        print("="*70)
        print("🚀 Initializing Inference Pipeline")
        print("="*70)
        
        # Load trained ControlNet
        print(f"📥 Loading ControlNet from: {controlnet_path}")
        if os.path.isfile(controlnet_path):
            # Load from .pt checkpoint
            print("   Loading from .pt checkpoint...")
            # First create a controlnet from config
            from diffusers import StableDiffusionXLPipeline
            temp_pipe = StableDiffusionXLPipeline.from_pretrained(
                base_model,
                torch_dtype=torch_dtype
            )
            controlnet = ControlNetModel.from_unet(
                temp_pipe.unet,
                conditioning_channels=4
            )
            # Load state dict
            state_dict = torch.load(controlnet_path, map_location=device)
            controlnet.load_state_dict(state_dict)
            del temp_pipe
        else:
            # Load from diffusers format directory
            print("   Loading from diffusers format...")
            controlnet = ControlNetModel.from_pretrained(
                controlnet_path,
                torch_dtype=torch_dtype
            )
        
        print(f"   ✓ ControlNet loaded (conditioning channels: {controlnet.config.conditioning_channels})")
        
        # Load base SDXL pipeline with ControlNet
        print(f"📥 Loading SDXL base model: {base_model}")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            force_zeros_for_empty_prompt=True
        )
        
        # Use DDIM scheduler if requested
        if use_ddim:
            print("🔧 Using DDIM scheduler")
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        else:
            print(f"🔧 Using {self.pipe.scheduler.__class__.__name__} scheduler")
        
        # Move to device
        self.pipe = self.pipe.to(device)
        
        # Set to eval mode for inference
        self.pipe.unet.eval()
        self.pipe.controlnet.eval()
        
        # Enable memory efficient attention if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ Enabled xformers memory efficient attention")
        except:
            print("⚠ xformers not available, using standard attention")
        
        print(f"✅ Pipeline ready on {device}")
        print("="*70 + "\n")
    
    def preprocess_inputs(self, mask_path, cond_path, size=512, invert_mask=False):
        """
        Load and preprocess mask and conditioning image.
        
        Args:
            mask_path: Path to binary mask image
            cond_path: Path to conditioning (masked) image
            size: Target size for resizing
            invert_mask: Whether to invert mask (swap 0s and 1s)
            
        Returns:
            tuple: (mask_tensor, cond_tensor, original_size)
        """
        # Load images
        mask_img = Image.open(mask_path).convert("L")  # Grayscale
        cond_img = Image.open(cond_path).convert("RGB")
        
        # Store original size
        original_size = mask_img.size  # (width, height)
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        
        # Apply transforms
        mask = transform(mask_img)  # [1, H, W]
        cond = transform(cond_img)  # [3, H, W]
        
        # Invert mask if needed (swap hole/valid regions)
        if invert_mask:
            mask = 1.0 - mask
        
        # Add batch dimension
        mask = mask.unsqueeze(0)  # [1, 1, H, W]
        cond = cond.unsqueeze(0)  # [1, 3, H, W]
        
        return mask, cond, original_size
    
    def inpaint(
        self,
        mask,
        cond,
        prompt="",
        negative_prompt="",
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=None,
        return_latents=False
    ):
        """
        Perform inpainting inference.
        
        Args:
            mask: Mask tensor [1, 1, H, W] where 1=hole to inpaint, 0=valid region
            cond: Conditioning image tensor [1, 3, H, W]
            prompt: Text prompt (typically empty for pure mask-based inpainting)
            negative_prompt: Negative text prompt
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility (None for random)
            return_latents: Whether to return intermediate latents
            
        Returns:
            PIL.Image or tuple: Generated inpainted image (and latents if requested)
        """
        # Move inputs to device
        mask = mask.to(self.device)
        cond = cond.to(self.device)
        
        # Prepare conditioning: set holes (mask=1) to white in conditioning image
        # This matches the training setup where holes are set to white
        cond_processed = torch.where(mask == 1, torch.ones_like(cond), cond)
        
        # Concatenate mask + processed conditioning (4 channels total)
        controlnet_cond = torch.cat([mask, cond_processed], dim=1)  # [1, 4, H, W]
        
        # Set random seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Run inference
        # Note: For StableDiffusionXLControlNetPipeline, the 'image' parameter
        # is what gets passed to the ControlNet, so it needs to be 4-channel
        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=controlnet_cond,  # 4-channel: mask (1) + masked RGB image (3)
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                height=mask.shape[2],
                width=mask.shape[3],
                output_type="pil" if not return_latents else "latent"
            )
        
        if return_latents:
            return output
        else:
            return output.images[0]
    
    def inpaint_single(
        self,
        mask_path,
        cond_path,
        output_path,
        size=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42,
        invert_mask=False,
        save_inputs=False
    ):
        """
        Perform inpainting on a single image pair.
        
        Args:
            mask_path: Path to mask image
            cond_path: Path to conditioning image
            output_path: Path to save result
            size: Processing size
            num_inference_steps: Number of diffusion steps
            guidance_scale: CFG scale
            seed: Random seed
            invert_mask: Whether to invert mask
            save_inputs: Whether to save preprocessed inputs alongside result
        """
        print("\n" + "="*70)
        print("🎯 Single Image Inpainting")
        print("="*70)
        print(f"Input mask:        {mask_path}")
        print(f"Input conditioning: {cond_path}")
        print(f"Output:            {output_path}")
        print(f"Parameters:")
        print(f"  - Size: {size}x{size}")
        print(f"  - Steps: {num_inference_steps}")
        print(f"  - Guidance: {guidance_scale}")
        print(f"  - Seed: {seed}")
        print("-"*70)
        
        # Preprocess inputs
        mask, cond, original_size = self.preprocess_inputs(
            mask_path, cond_path, size=size, invert_mask=invert_mask
        )
        print(f"✓ Loaded and preprocessed inputs (original size: {original_size})")
        
        # Generate
        print("🎨 Generating inpainted image...")
        result = self.inpaint(
            mask=mask,
            cond=cond,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed
        )
        print("✓ Generation complete!")
        
        # Resize back to original size if needed
        if original_size != (size, size):
            result = result.resize(original_size, Image.LANCZOS)
            print(f"✓ Resized to original size: {original_size}")
        
        # Save result
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        result.save(output_path)
        print(f"💾 Saved result to: {output_path}")
        
        # Optionally save preprocessed inputs for debugging
        if save_inputs:
            input_dir = os.path.join(os.path.dirname(output_path), "debug_inputs")
            os.makedirs(input_dir, exist_ok=True)
            
            mask_img = Image.fromarray((mask.squeeze().cpu().numpy() * 255).astype(np.uint8))
            mask_img.save(os.path.join(input_dir, "mask_preprocessed.png"))
            
            cond_img = (cond.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(cond_img).save(os.path.join(input_dir, "cond_preprocessed.png"))
            
            print(f"💾 Saved debug inputs to: {input_dir}")
        
        print("="*70)
        return result
    
    def inpaint_batch(
        self,
        input_dir,
        output_dir,
        size=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42,
        invert_mask=False,
        save_comparison=True
    ):
        """
        Run batch inference on a directory structure.
        
        Expected structure:
            input_dir/
                scene1/
                    pair1/
                        mask_cond.png
                        image_cond.png
                        image_target.png (optional, for comparison)
                    pair2/
                        ...
                scene2/
                    ...
        
        Args:
            input_dir: Root directory with scene/pair structure
            output_dir: Output directory for results
            size: Processing size
            num_inference_steps: Number of diffusion steps
            guidance_scale: CFG scale
            seed: Random seed
            invert_mask: Whether to invert masks
            save_comparison: Whether to save side-by-side comparisons
        """
        print("\n" + "="*70)
        print("🔄 Batch Inference Mode")
        print("="*70)
        print(f"Input directory:  {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Parameters:")
        print(f"  - Size: {size}x{size}")
        print(f"  - Steps: {num_inference_steps}")
        print(f"  - Guidance: {guidance_scale}")
        print(f"  - Seed: {seed}")
        print("-"*70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect all samples
        samples = []
        for scene_dir in sorted(os.listdir(input_dir)):
            scene_path = os.path.join(input_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue
            
            for pair_dir in sorted(os.listdir(scene_path)):
                pair_path = os.path.join(scene_path, pair_dir)
                if not os.path.isdir(pair_path):
                    continue
                
                # Try different naming conventions
                mask_path = os.path.join(pair_path, "sample_mask_cond.png")
                if not os.path.exists(mask_path):
                    mask_path = os.path.join(pair_path, "mask_cond.png")
                
                cond_path = os.path.join(pair_path, "sample_image_cond.png")
                if not os.path.exists(cond_path):
                    cond_path = os.path.join(pair_path, "image_cond.png")
                
                target_path = os.path.join(pair_path, "sample_image_target.png")
                if not os.path.exists(target_path):
                    target_path = os.path.join(pair_path, "image_target.png")
                
                if os.path.exists(mask_path) and os.path.exists(cond_path):
                    samples.append({
                        "mask_path": mask_path,
                        "cond_path": cond_path,
                        "target_path": target_path if os.path.exists(target_path) else None,
                        "scene": scene_dir,
                        "pair": pair_dir
                    })
        
        print(f"✅ Found {len(samples)} samples to process\n")
        
        # Process each sample
        successful = 0
        failed = 0
        
        for idx, sample in enumerate(tqdm(samples, desc="Processing")):
            try:
                # Preprocess inputs
                mask, cond, original_size = self.preprocess_inputs(
                    sample["mask_path"],
                    sample["cond_path"],
                    size=size,
                    invert_mask=invert_mask
                )
                
                # Generate
                result = self.inpaint(
                    mask=mask,
                    cond=cond,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                
                # Resize back to original size if needed
                if original_size != (size, size):
                    result = result.resize(original_size, Image.LANCZOS)
                
                # Save result
                scene_output_dir = os.path.join(output_dir, sample["scene"], sample["pair"])
                os.makedirs(scene_output_dir, exist_ok=True)
                
                output_path = os.path.join(scene_output_dir, "inpainted.png")
                result.save(output_path)
                
                # Optionally save comparison
                if save_comparison and sample["target_path"]:
                    self._save_comparison(
                        mask=mask,
                        cond=cond,
                        result=result,
                        target_path=sample["target_path"],
                        output_dir=scene_output_dir,
                        original_size=original_size
                    )
                
                successful += 1
                
            except Exception as e:
                failed += 1
                print(f"\n❌ Error processing {sample['scene']}/{sample['pair']}: {e}")
                continue
        
        print("\n" + "="*70)
        print(f"✅ Batch inference complete!")
        print(f"   Successful: {successful}/{len(samples)}")
        print(f"   Failed: {failed}/{len(samples)}")
        print(f"   Results saved to: {output_dir}")
        print("="*70)
    
    def _save_comparison(self, mask, cond, result, target_path, output_dir, original_size):
        """Save side-by-side comparison of inputs and outputs."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Load target image
        target = Image.open(target_path).convert("RGB")
        if target.size != original_size:
            target = target.resize(original_size, Image.LANCZOS)
        
        # Resize result if needed
        if result.size != original_size:
            result = result.resize(original_size, Image.LANCZOS)
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Mask
        mask_np = mask.squeeze().cpu().numpy()
        axes[0].imshow(mask_np, cmap='gray')
        axes[0].set_title("Input Mask", fontsize=14)
        axes[0].axis('off')
        
        # Conditioning (with white holes)
        cond_with_holes = torch.where(mask == 1, torch.ones_like(cond), cond)
        cond_np = cond_with_holes.squeeze().permute(1, 2, 0).cpu().numpy()
        axes[1].imshow(np.clip(cond_np, 0, 1))
        axes[1].set_title("Input (Masked)", fontsize=14)
        axes[1].axis('off')
        
        # Generated
        axes[2].imshow(result)
        axes[2].set_title("Generated", fontsize=14)
        axes[2].axis('off')
        
        # Ground Truth
        axes[3].imshow(target)
        axes[3].set_title("Ground Truth", fontsize=14)
        axes[3].axis('off')
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "comparison.png")
        plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Inference for ControlNet-based Inpainting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python inference.py --controlnet_path output/controlnet_final \\
      --mask_path data/test/scene1/pair1/mask_cond.png \\
      --cond_path data/test/scene1/pair1/image_cond.png \\
      --output_path output/result.png

  # Batch inference
  python inference.py --controlnet_path output/controlnet_final \\
      --input_dir data/test \\
      --output_dir output/results \\
      --batch
        """
    )
    
    # Model settings
    parser.add_argument("--controlnet_path", type=str, required=True,
                       help="Path to trained ControlNet (diffusers format or .pt file)")
    parser.add_argument("--base_model", type=str, 
                       default="stabilityai/stable-diffusion-xl-base-1.0",
                       help="Base SDXL model")
    
    # Mode selection
    parser.add_argument("--batch", action="store_true",
                       help="Run batch inference on directory")
    
    # Single inference
    parser.add_argument("--mask_path", type=str,
                       help="Path to mask image (single inference)")
    parser.add_argument("--cond_path", type=str,
                       help="Path to conditioning image (single inference)")
    parser.add_argument("--output_path", type=str,
                       help="Output path (single inference)")
    
    # Batch inference
    parser.add_argument("--input_dir", type=str,
                       help="Input directory (batch inference)")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory (batch inference)")
    
    # Inference parameters
    parser.add_argument("--size", type=int, default=512,
                       help="Image size for processing")
    parser.add_argument("--num_steps", type=int, default=50,
                       help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    # Options
    parser.add_argument("--invert_mask", action="store_true",
                       help="Invert mask (swap holes and valid regions)")
    parser.add_argument("--use_ddim", action="store_true", default=True,
                       help="Use DDIM scheduler")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to run on")
    parser.add_argument("--fp16", action="store_true",
                       help="Use float16 precision (faster but less accurate)")
    parser.add_argument("--save_inputs", action="store_true",
                       help="Save preprocessed inputs (debugging)")
    parser.add_argument("--save_comparison", action="store_true", default=True,
                       help="Save comparison images in batch mode")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.batch:
        if not args.input_dir or not args.output_dir:
            parser.error("Batch mode requires --input_dir and --output_dir")
    else:
        if not args.mask_path or not args.cond_path or not args.output_path:
            parser.error("Single mode requires --mask_path, --cond_path, and --output_path")
    
    # Set dtype
    torch_dtype = torch.float16 if args.fp16 else torch.float32
    
    # Initialize inference pipeline
    inferencer = InpaintingInference(
        controlnet_path=args.controlnet_path,
        base_model=args.base_model,
        device=args.device,
        use_ddim=args.use_ddim,
        torch_dtype=torch_dtype
    )
    
    # Run inference
    if args.batch:
        inferencer.inpaint_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            size=args.size,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            invert_mask=args.invert_mask,
            save_comparison=args.save_comparison
        )
    else:
        inferencer.inpaint_single(
            mask_path=args.mask_path,
            cond_path=args.cond_path,
            output_path=args.output_path,
            size=args.size,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            invert_mask=args.invert_mask,
            save_inputs=args.save_inputs
        )


if __name__ == "__main__":
    main()
