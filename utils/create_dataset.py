import os
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import cv2

# === Import your geometry and rendering utilities ===
from .warp_utils import (
    depth_to_points,
    apply_cam_yaw, 
    rasterize_points_world,
    project_invalid
)

def generate_inpaint_pair(image_path, depth_path, K, T_src_w2c, yaw_deg, valid_thresh=0.3):
    """
    Generate one (image_cond, mask_cond, image_target) triplet.
    """
    rgb = read_image(image_path).float() / 255.0
    if rgb.shape[0] == 1:
        rgb = rgb.repeat(3,1,1)
    src_rgb = rgb.permute(1,2,0).numpy()
    depth = np.load(depth_path).astype(np.float32)
    H, W = depth.shape

    # 1. Convert depth → world points directly using Open3D
    pts_world, cols = depth_to_points(depth, src_rgb, K, T_src_w2c)

    # 2. Create target camera pose with small yaw
    T_tgt_w2c = apply_cam_yaw(T_src_w2c, yaw_deg)

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


def process_scene(scene_dir, out_dir, num_pairs=5, yaw_range=10.0):
    """
    Create N inpainting pairs for one scene.
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
    count = 0

    for i in range(num_pairs):
        yaw = np.random.uniform(-yaw_range, yaw_range)
        result = generate_inpaint_pair(image_path, depth_path, K, T_src_w2c, yaw)
        if result is None:
            continue
        image_cond, mask_cond, image_target = result
        pair_dir = os.path.join(out_dir, f"pair_{count:03d}")
        os.makedirs(pair_dir, exist_ok=True)
        save_image(image_target, os.path.join(pair_dir, "image_target.png"))
        save_image(image_cond, os.path.join(pair_dir, "image_cond.png"))
        save_image(mask_cond, os.path.join(pair_dir, "mask_cond.png"))
        count += 1

    print(f"Saved {count} valid pairs for {os.path.basename(scene_dir)}")


def main():
    parser = argparse.ArgumentParser("Create forward–backward inpainting dataset")
    parser.add_argument("--data", required=True, help="Input data root (with scenes)")
    parser.add_argument("--out", required=True, help="Output dataset root")
    parser.add_argument("--pairs_per_scene", type=int, default=5)
    parser.add_argument("--yaw_range", type=float, default=15.0)
    args = parser.parse_args()

    scenes = [os.path.join(args.data, d) for d in sorted(os.listdir(args.data)) if os.path.isdir(os.path.join(args.data, d))]
    print(f"Found {len(scenes)} scenes.")

    for scene_dir in tqdm(scenes):
        out_dir = os.path.join(args.out, os.path.basename(scene_dir))
        process_scene(scene_dir, out_dir, args.pairs_per_scene, args.yaw_range)


if __name__ == "__main__":
    main()
