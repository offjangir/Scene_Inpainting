import os
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import cv2
import itertools

# === Import your geometry and rendering utilities ===
from .warp_utils import (
    depth_to_points,
    apply_cam_yaw,
    apply_cam_transform,
    rasterize_points_world,
    project_invalid
)

def generate_inpaint_pair(image_path, depth_path, K, T_src_w2c, 
                         yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                         translate_x=0.0, translate_y=0.0, translate_z=0.0,
                         valid_thresh=0.5):
    """
    Generate one (image_cond, mask_cond, image_target) triplet.
    
    Args:
        image_path: Path to source RGB image
        depth_path: Path to depth map (.npy)
        K: Camera intrinsics (3x3)
        T_src_w2c: Source world-to-camera transform (4x4)
        yaw_deg: Yaw rotation in degrees
        pitch_deg: Pitch rotation in degrees
        roll_deg: Roll rotation in degrees
        translate_x: Translation along camera X axis
        translate_y: Translation along camera Y axis
        translate_z: Translation along camera Z axis
        valid_thresh: Minimum valid pixel ratio threshold
    """
    rgb = read_image(image_path).float() / 255.0
    if rgb.shape[0] == 1:
        rgb = rgb.repeat(3,1,1)
    src_rgb = rgb.permute(1,2,0).numpy()
    depth = np.load(depth_path).astype(np.float32)
    H, W = depth.shape

    # 1. Convert depth → world points directly using Open3D
    pts_world, cols = depth_to_points(depth, src_rgb, K, T_src_w2c)

    # 2. Create target camera pose with full 6-DOF transformation
    T_tgt_w2c = apply_cam_transform(T_src_w2c, yaw_deg, pitch_deg, roll_deg,
                                   translate_x, translate_y, translate_z)

    # 3. Forward–backward rasterization
    # img_fwd, mask_fwd = rasterize_points_world(pts_world, cols, K, T_src_w2c, W, H)
    # img_bwd, mask_bwd = rasterize_points_world(pts_world, cols, K, T_tgt_w2c, W, H)
    # # save img_bwd for visualization
    # cv2.imwrite("debug_bwd.png", (img_bwd[..., ::-1] * 255).astype(np.uint8))
    # cv2.imwrite("debug_fwd.png", (img_fwd[..., ::-1] * 255).astype(np.uint8))
    # breakpoint()
    mask_consistent = project_invalid(pts_world, cols, K, T_tgt_w2c, W, H).reshape(H, W)

    # 4. Valid & hole masks
    # mask_consistent = (mask_fwd > 0) & (mask_bwd > 0)
    valid_ratio = mask_consistent.mean()
    print("Valid ratio:", valid_ratio)
    if valid_ratio < valid_thresh:
        return None  # skip if too much disocclusion

    mask_cond = (1 - mask_consistent.astype(np.float32))
    image_target = torch.from_numpy(src_rgb).permute(2,0,1)
    image_cond = torch.from_numpy(src_rgb * (1 - mask_cond[...,None])).permute(2,0,1)
    mask_tensor = torch.from_numpy(mask_cond).unsqueeze(0)

    return image_cond, mask_tensor, image_target


def process_scene(scene_dir, out_dir, num_pairs=5, 
                 yaw_range=10.0, pitch_range=0.0, roll_range=0.0,
                 translate_x_range=0.0, translate_y_range=0.0, translate_z_range=0.0,
                 num_steps_per_param=3):
    """
    Create N inpainting pairs for one scene with systematic 6-DOF perturbations.
    
    Strategy:
    1. Generate systematic grid of perturbations for each parameter
    2. Create all combinations or sample evenly from them
    
    Args:
        scene_dir: Path to scene directory
        out_dir: Output directory for pairs
        num_pairs: Number of pairs to generate (e.g., 10)
        yaw_range: Max yaw rotation in degrees (+/-)
        pitch_range: Max pitch rotation in degrees (+/-)
        roll_range: Max roll rotation in degrees (+/-)
        translate_x_range: Max X translation (+/-)
        translate_y_range: Max Y translation (+/-)
        translate_z_range: Max Z translation (+/-)
        num_steps_per_param: Number of steps for each parameter (e.g., 3 means [-range, 0, +range])
    """
    image_path = os.path.join(scene_dir, "images0/0.jpg")
    depth_path = os.path.join(scene_dir, "depth/0.npy")
    K_path = os.path.join(scene_dir, "intrinsics.npy")
    T_path = os.path.join(scene_dir, "extrinsics.npy")

    if not all(os.path.exists(p) for p in [image_path, depth_path, K_path, T_path]):
        print(f"Skipping {scene_dir} (missing files)")
        return

    K = np.load(K_path).astype(np.float32)
    T_src_w2c = np.load(T_path).astype(np.float32)
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Generate systematic perturbation values for each parameter
    print(f"  Generating systematic perturbations with {num_steps_per_param} steps per parameter...")
    
    def create_linspace(param_range, num_steps):
        """Create evenly spaced values, excluding 0 if range is 0"""
        if param_range == 0.0:
            return [0.0]
        else:
            return np.linspace(-param_range, param_range, num_steps).tolist()
    
    yaw_values = create_linspace(yaw_range, num_steps_per_param)
    pitch_values = create_linspace(pitch_range, num_steps_per_param)
    roll_values = create_linspace(roll_range, num_steps_per_param)
    tx_values = create_linspace(translate_x_range, num_steps_per_param)
    ty_values = create_linspace(translate_y_range, num_steps_per_param)
    tz_values = create_linspace(translate_z_range, num_steps_per_param)
    
    # Step 2: Create all combinations systematically
    all_combinations = list(itertools.product(
        yaw_values, pitch_values, roll_values,
        tx_values, ty_values, tz_values
    ))
    
    # Remove the identity transformation (all zeros)
    all_combinations = [c for c in all_combinations if any(abs(v) > 1e-6 for v in c)]
    
    print(f"  Generated {len(all_combinations)} systematic pose combinations")
    
    # Step 3: Sample evenly from systematic combinations if needed
    if len(all_combinations) > num_pairs:
        # Sample evenly spaced indices
        indices = np.linspace(0, len(all_combinations)-1, num_pairs, dtype=int)
        sampled_combinations = [all_combinations[i] for i in indices]
    else:
        sampled_combinations = all_combinations
    
    sampled_poses = [
        {
            'yaw': c[0], 'pitch': c[1], 'roll': c[2],
            'tx': c[3], 'ty': c[4], 'tz': c[5]
        }
        for c in sampled_combinations
    ]
    
    print(f"  Selected {len(sampled_poses)} systematic poses")
    
    # Step 3: Generate pairs from sampled poses
    count = 0
    for pose in sampled_poses:
        result = generate_inpaint_pair(
            image_path, depth_path, K, T_src_w2c,
            yaw_deg=pose['yaw'],
            pitch_deg=pose['pitch'],
            roll_deg=pose['roll'],
            translate_x=pose['tx'],
            translate_y=pose['ty'],
            translate_z=pose['tz']
        )
        
        if result is None:
            print(f"⚠️ Skipping invalid pose (yaw={pose['yaw']:.1f}°)")
            continue
            
        image_cond, mask_cond, image_target = result
        pair_dir = os.path.join(out_dir, f"pair_{count:03d}")
        os.makedirs(pair_dir, exist_ok=True)
        
        save_image(image_target, os.path.join(pair_dir, "image_target.png"))
        save_image(image_cond, os.path.join(pair_dir, "image_cond.png"))
        save_image(mask_cond, os.path.join(pair_dir, "mask_cond.png"))
        
        # Save pose parameters
        with open(os.path.join(pair_dir, "params.txt"), 'w') as f:
            f.write(f"Yaw: {pose['yaw']:.3f}°\n")
            f.write(f"Pitch: {pose['pitch']:.3f}°\n")
            f.write(f"Roll: {pose['roll']:.3f}°\n")
            f.write(f"Tx: {pose['tx']:.3f}\n")
            f.write(f"Ty: {pose['ty']:.3f}\n")
            f.write(f"Tz: {pose['tz']:.3f}\n")
        
        count += 1

    print(f"  ✅ Saved {count} valid pairs for {os.path.basename(scene_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create forward–backward inpainting dataset with systematic 6-DOF camera perturbations"
    )
    # IO arguments
    parser.add_argument("--data", required=True, help="Input data root (with scenes)")
    parser.add_argument("--out", required=True, help="Output dataset root")
    parser.add_argument("--pairs_per_scene", type=int, default=2, 
                       help="Number of pairs per scene")
    
    # Rotation perturbations (in degrees)
    parser.add_argument("--yaw_range", type=float, default=8.0,
                       help="Max yaw rotation range in degrees (+/-)")
    parser.add_argument("--pitch_range", type=float, default=0.0,
                       help="Max pitch rotation range in degrees (+/-)")
    parser.add_argument("--roll_range", type=float, default=0.0,
                       help="Max roll rotation range in degrees (+/-)")
    
    # Translation perturbations (in scene units, typically meters)
    parser.add_argument("--translate_x_range", type=float, default=0.0,
                       help="Max X translation range (+/-)")
    parser.add_argument("--translate_y_range", type=float, default=0.0,
                       help="Max Y translation range (+/-)")
    parser.add_argument("--translate_z_range", type=float, default=0.0,
                       help="Max Z translation range (+/-)")
    
    # Sampling strategy
    parser.add_argument("--num_steps_per_param", type=int, default=3,
                       help="Number of systematic steps per parameter (e.g., 3 = [-range, 0, +range])")
    
    args = parser.parse_args()

    scenes = [os.path.join(args.data, d) for d in sorted(os.listdir(args.data)) 
              if os.path.isdir(os.path.join(args.data, d))]
    print(f"Found {len(scenes)} scenes.")
    print(f"Perturbation ranges:")
    print(f"  Yaw: ±{args.yaw_range}°, Pitch: ±{args.pitch_range}°, Roll: ±{args.roll_range}°")
    print(f"  Translation X: ±{args.translate_x_range}, Y: ±{args.translate_y_range}, Z: ±{args.translate_z_range}")
    print(f"Sampling strategy: Systematic grid with {args.num_steps_per_param} steps per parameter")

    for scene_dir in tqdm(scenes):
        out_dir = os.path.join(args.out, os.path.basename(scene_dir))
        process_scene(
            scene_dir, out_dir, args.pairs_per_scene, 
            args.yaw_range, args.pitch_range, args.roll_range,
            args.translate_x_range, args.translate_y_range, args.translate_z_range,
            args.num_steps_per_param
        )


if __name__ == "__main__":
    main()
