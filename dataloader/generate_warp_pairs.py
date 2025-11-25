import os
import sys
import argparse
import numpy as np
import torch
from torchvision.io import read_image, write_png
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# Add parent directory to path to import from utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import utility functions (using PyTorch3D-based rasterization)
from utils.warp_utils import (
    depth_to_points,
    apply_cam_yaw,
    apply_cam_transform,
    rasterize_points_world,
    project_invalid
)


def generate_warp_pair(
    image_path, depth_path, K_path, T_src_w2c_path,
    yaw_deg=30.0, dx=0.0, dy=0.0, dz=0.0,
    out_dir="out",
    basename="sample"
):
    """
    Generate training pair:
    1. image_target.png - original view (ground truth)
    2. image_cond.png - warped view with holes (conditioning input)
    3. mask_cond.png - binary mask (white=holes to inpaint, black=valid pixels)
    
    Args:
        image_path: Path to RGB image
        depth_path: Path to depth map (.npy)
        K_path: Path to intrinsics matrix (.npy)
        T_src_w2c_path: Path to source extrinsics (world→cam)
        yaw_deg: Yaw rotation in degrees
        dx, dy, dz: Translation in camera frame (right, up, forward)
        out_dir: Output directory
        basename: Base name for output files
    
    Returns:
        dict with paths to saved files
    """
    
    # ============================================================
    # Load data (same approach as create_dataset.py)
    # ============================================================
    K = np.load(K_path).astype(np.float32)
    T_src_w2c = np.load(T_src_w2c_path).astype(np.float32)
    
    # Handle (3,4) extrinsics
    if T_src_w2c.ndim == 2 and T_src_w2c.shape[0] == 3:
        E = np.eye(4, dtype=np.float32)
        E[:3, :4] = T_src_w2c
        T_src_w2c = E
    elif T_src_w2c.ndim == 3:
        T_src_w2c = T_src_w2c[0]
        if T_src_w2c.shape == (3, 4):
            E = np.eye(4, dtype=np.float32)
            E[:3, :4] = T_src_w2c
            T_src_w2c = E
    
    # Load RGB and depth
    rgb = read_image(image_path).float() / 255.0  # (3,H,W)
    if rgb.shape[0] == 1:
        rgb = rgb.repeat(3, 1, 1)
    src_rgb = rgb.permute(1, 2, 0).numpy()  # (H,W,3)
    depth = np.load(depth_path).astype(np.float32)
    H, W = depth.shape
    
    # ============================================================
    # Step 1: Convert depth → world points using Open3D (same as create_dataset.py)
    # ============================================================
    pts_world, cols = depth_to_points(depth, src_rgb, K, T_src_w2c)
    
    # ============================================================
    # Step 2: Create target pose with 6-DOF transformation
    # ============================================================
    # Convert yaw rotation + translations to full 6-DOF
    # Note: dx, dy, dz are in camera frame, need to convert to pitch/roll/tx/ty/tz
    T_tgt_w2c = apply_cam_transform(
        T_src_w2c, 
        yaw_deg=yaw_deg, 
        pitch_deg=0.0,  # Could be parameterized if needed
        roll_deg=0.0,   # Could be parameterized if needed
        translate_x=dx, 
        translate_y=dy, 
        translate_z=dz
    )
    
    # ============================================================
    # Step 3: Rasterize to get warped image and mask (PyTorch3D-based)
    # ============================================================
    img_warped, mask_warped = rasterize_points_world(
        pts_world, cols, K, T_tgt_w2c, W, H
    )
    
    # ============================================================
    # Step 4: Save outputs
    # ============================================================
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Save original image (target/ground truth)
    target_path = os.path.join(out_dir, f"{basename}_image_target.png")
    src_rgb_uint8 = (src_rgb * 255).astype(np.uint8)
    Image.fromarray(src_rgb_uint8).save(target_path)
    
    # 2. Save warped image (conditioning input with holes)
    cond_path = os.path.join(out_dir, f"{basename}_image_cond.png")
    img_warped_uint8 = (img_warped * 255).astype(np.uint8)
    Image.fromarray(img_warped_uint8).save(cond_path)
    
    # 3. Save mask (white=holes to inpaint, black=valid pixels)
    mask_path = os.path.join(out_dir, f"{basename}_mask_cond.png")
    # Invert mask: 1 where there are holes (no content), 0 where there is content
    mask_holes = (mask_warped == 0).astype(np.uint8) * 255
    Image.fromarray(mask_holes).save(mask_path)
    
    # ============================================================
    # Calculate statistics
    # ============================================================
    total_pixels = H * W
    valid_pixels = mask_warped.sum()
    hole_pixels = total_pixels - valid_pixels
    
    return {
        "target_path": target_path,
        "cond_path": cond_path,
        "mask_path": mask_path,
        "stats": {
            "total_pixels": total_pixels,
            "valid_pixels": int(valid_pixels),
            "hole_pixels": int(hole_pixels),
            "hole_percentage": 100 * hole_pixels / total_pixels
        }
    }


def process_all_scenes(data_dir, out_dir, yaw_range=30.0, num_pairs=10):
    """
    Process all scenes in a directory with systematic yaw perturbations.
    Creates warped test images for evaluation.
    
    Args:
        data_dir: Root directory containing scenes (e.g., dataset_gene)
        out_dir: Output directory for warped test pairs
        yaw_range: Max yaw rotation in degrees (+/-)
        num_pairs: Number of systematic yaw perturbations per scene
    """
    # Get all scene directories
    scenes = [d for d in sorted(os.listdir(data_dir)) 
              if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('scene')]
    
    print(f"\n{'='*60}")
    print(f"WARPED TEST SET GENERATION")
    print(f"{'='*60}")
    print(f"Found {len(scenes)} scenes in {data_dir}")
    print(f"Will generate {num_pairs} warped pairs per scene")
    print(f"Yaw range: -{yaw_range}° to +{yaw_range}°")
    
    # Generate systematic yaw values
    yaw_values = np.linspace(-yaw_range, yaw_range, num_pairs)
    print(f"Yaw values: {[f'{y:.1f}°' for y in yaw_values]}")
    print(f"Output directory: {out_dir}\n")
    
    total_pairs = 0
    failed_pairs = 0
    
    for scene_name in tqdm(scenes, desc="Processing scenes"):
        scene_dir = os.path.join(data_dir, scene_name)
        scene_out_dir = os.path.join(out_dir, scene_name)
        os.makedirs(scene_out_dir, exist_ok=True)
        
        # Check required files exist
        image_path = os.path.join(scene_dir, "images0/0.jpg")
        depth_path = os.path.join(scene_dir, "depth/0.npy")
        K_path = os.path.join(scene_dir, "intrinsics.npy")
        T_path = os.path.join(scene_dir, "extrinsics.npy")
        
        if not all(os.path.exists(p) for p in [image_path, depth_path, K_path, T_path]):
            print(f"\n⚠️  Skipping {scene_name} (missing files)")
            continue
        
        # Generate pairs for each yaw value
        for i, yaw_deg in enumerate(yaw_values):
            pair_dir = os.path.join(scene_out_dir, f"pair_{i}")
            
            try:
                result = generate_warp_pair(
                    image_path, depth_path, K_path, T_path,
                    yaw_deg=yaw_deg,
                    dx=0.0, dy=0.0, dz=0.0,
                    out_dir=pair_dir,
                    basename="sample"
                )
                total_pairs += 1
            except Exception as e:
                print(f"\n  ✗ Failed {scene_name}/pair_{i} (yaw={yaw_deg:.1f}°): {e}")
                failed_pairs += 1
    
    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"✅ Successfully generated: {total_pairs} pairs")
    if failed_pairs > 0:
        print(f"❌ Failed: {failed_pairs} pairs")
    print(f"📁 Output directory: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate warped test pairs for inpainting model evaluation"
    )
    
    # Mode selection
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                       help="Processing mode: 'single' for one image, 'batch' for all scenes")
    
    # Batch mode arguments
    parser.add_argument("--data_dir", help="[Batch mode] Root directory containing scenes")
    parser.add_argument("--yaw_range", type=float, default=30.0,
                       help="[Batch mode] Max yaw rotation in degrees (+/-)")
    parser.add_argument("--num_pairs", type=int, default=10,
                       help="[Batch mode] Number of systematic yaw perturbations")
    
    # Single mode arguments (kept for backward compatibility)
    parser.add_argument("--image", help="[Single mode] Path to RGB image")
    parser.add_argument("--depth", help="[Single mode] Path to depth map (.npy)")
    parser.add_argument("--K", help="[Single mode] Path to intrinsics matrix (.npy)")
    parser.add_argument("--T_src_w2c", help="[Single mode] Path to extrinsics (.npy)")
    parser.add_argument("--yaw", type=float, default=30.0, 
                       help="[Single mode] Yaw rotation in degrees")
    parser.add_argument("--dx", type=float, default=0.0, 
                       help="[Single mode] Translation right (+) / left (-) in meters")
    parser.add_argument("--dy", type=float, default=0.0, 
                       help="[Single mode] Translation up (+) / down (-) in meters")
    parser.add_argument("--dz", type=float, default=0.0, 
                       help="[Single mode] Translation forward (+) / backward (-) in meters")
    parser.add_argument("--basename", default="sample", 
                       help="[Single mode] Base name for output files")
    
    # Common arguments
    parser.add_argument("--out_dir", help="Output directory")
    
    args = parser.parse_args()
    
    if args.mode == "batch":
        # Batch processing mode
        if not args.data_dir:
            parser.error("--data_dir is required for batch mode")
        if not args.out_dir:
            parser.error("--out_dir is required for batch mode")
        
        process_all_scenes(
            args.data_dir, 
            args.out_dir, 
            yaw_range=args.yaw_range,
            num_pairs=args.num_pairs
        )
    
    else:
        # Single file mode
        if not all([args.image, args.depth, args.K, args.T_src_w2c]):
            parser.error("--image, --depth, --K, and --T_src_w2c are required for single mode")
        if not args.out_dir:
            args.out_dir = "out"
        
        result = generate_warp_pair(
            args.image, args.depth, args.K, args.T_src_w2c,
            yaw_deg=args.yaw, dx=args.dx, dy=args.dy, dz=args.dz,
            out_dir=args.out_dir,
            basename=args.basename
        )

