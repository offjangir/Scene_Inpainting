import os
import argparse
import numpy as np
import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt

# Import utility functions
from viz_warp_np import (
    depth_to_points_cam, 
    cam2world_from_world2cam,
    apply_cam_yaw, 
    rasterize_points_world,
    overlay_mask,
    make_grid
)


def forward_backward_warp(
    image_path, depth_path, K_path, T_src_w2c_path,
    yaw_deg=30.0, dx=0.0, dy=0.0, dz=0.0,
    out_path="out/forward_backward_warp.png"
):
    """
    Forward-backward warping:
    1. Warp original image to new pose (forward) -> creates warped image with holes
    2. Use ONLY the visible points from the warped image to warp back to original pose
    
    Args:
        image_path: Path to RGB image
        depth_path: Path to depth map (.npy)
        K_path: Path to intrinsics matrix
        T_src_w2c_path: Path to source extrinsics (world→cam)
        yaw_deg: Yaw rotation in degrees
        dx, dy, dz: Translation in camera frame (right, up, forward)
        out_path: Output visualization path
    """
    
    # ============================================================
    # Load data
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
    depth = np.load(depth_path).astype(np.float32)
    
    H, W = rgb.shape[-2], rgb.shape[-1]
    src_rgb = rgb.permute(1, 2, 0).numpy()  # (H,W,3)
    
    # ============================================================
    # Step 1: Build world points from source view
    # ============================================================
    pts_cam, _ = depth_to_points_cam(depth, K)  # (H*W, 3) in camera frame
    T_src_c2w = cam2world_from_world2cam(T_src_w2c)
    pts_world = (T_src_c2w[:3, :3] @ pts_cam.T + T_src_c2w[:3, 3:4]).T  # (H*W, 3)
    cols = src_rgb.reshape(-1, 3)  # (H*W, 3)
    
    # ============================================================
    # Step 2: Create target pose (forward warp destination)
    # ============================================================
    T_tgt_w2c = apply_cam_yaw(T_src_w2c, yaw_deg)
    
    # Apply translation if specified
    if dx != 0.0 or dy != 0.0 or dz != 0.0:
        T_tgt_w2c = translate_camera(T_tgt_w2c, dx, dy, dz)
    
    # ============================================================
    # Step 3: Forward warp (src → tgt)
    # ============================================================
    print(f"Forward warping from source to target (yaw={yaw_deg}°, dx={dx}m)...")
    img_forward, mask_forward = rasterize_points_world(
        pts_world, cols, K, T_tgt_w2c, W, H
    )
    
    # ============================================================
    # Step 4: Extract ONLY visible points from forward warp
    # ============================================================
    # For each pixel in the forward-warped image that has content,
    # we need to find the corresponding 3D point
    
    # Re-project: which 3D points landed in the forward warped image?
    # We'll track this by transforming pts_world to target camera and checking visibility
    
    pts_h = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)
    pts_tgt_cam = (T_tgt_w2c @ pts_h.T).T  # (N, 4)
    z_tgt = pts_tgt_cam[:, 2]
    
    # Project to target image
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u_tgt = fx * (pts_tgt_cam[:, 0] / z_tgt) + cx
    v_tgt = fy * (pts_tgt_cam[:, 1] / z_tgt) + cy
    
    # Filter: only keep points that are visible in forward warp
    valid_tgt = (
        (z_tgt > 1e-4) & 
        (u_tgt >= 0) & (u_tgt < W) & 
        (v_tgt >= 0) & (v_tgt < H)
    )
    
    u_tgt_int = u_tgt[valid_tgt].astype(np.int32)
    v_tgt_int = v_tgt[valid_tgt].astype(np.int32)
    
    # Check which of these points actually rendered in the forward pass
    # (i.e., were not occluded by closer points)
    pts_visible_mask = mask_forward[v_tgt_int, u_tgt_int] > 0
    
    # Get indices of visible points in original point cloud
    valid_indices = np.where(valid_tgt)[0]
    visible_indices = valid_indices[pts_visible_mask]
    
    # Extract visible points and colors
    pts_world_visible = pts_world[visible_indices]
    cols_visible = cols[visible_indices]
    
    print(f"  Total points: {len(pts_world)}")
    print(f"  Visible in forward warp: {len(pts_world_visible)} ({100*len(pts_world_visible)/len(pts_world):.1f}%)")
    
    # ============================================================
    # Step 5: Backward warp (tgt → src) using ONLY visible points
    # ============================================================
    print(f"Backward warping from target back to source using only visible points...")
    img_backward, mask_backward = rasterize_points_world(
        pts_world_visible, cols_visible, K, T_src_w2c, W, H
    )
    
    # ============================================================
    # Step 6: Analyze reconstruction
    # ============================================================
    # Render original view with all points for comparison
    img_src_full, mask_src_full = rasterize_points_world(
        pts_world, cols, K, T_src_w2c, W, H
    )
    
    # Difference: which parts couldn't be reconstructed?
    mask_reconstructed = mask_backward > 0
    mask_lost = (mask_src_full > 0) & (~mask_reconstructed)
    
    # Overlays
    overlay_forward = overlay_mask(img_forward, mask_forward > 0, color=(0, 1, 0), alpha=0.2)
    overlay_backward = overlay_mask(img_backward, mask_backward > 0, color=(0, 0, 1), alpha=0.2)
    overlay_lost = overlay_mask(src_rgb, mask_lost, color=(1, 0, 0), alpha=0.5)
    overlay_reconstructed = overlay_mask(src_rgb, mask_reconstructed, color=(0, 1, 0), alpha=0.3)
    
    # ============================================================
    # Step 7: Visualize
    # ============================================================
    titles = [
        "1. Original Image",
        "2. Forward Warp (target view)",
        "3. Backward Warp (reconstructed)",
        "4. Reconstruction Mask",
        "5. Lost Regions (red)",
        "6. Forward Warp Mask"
    ]
    
    imgs = [
        torch.from_numpy(src_rgb).permute(2, 0, 1),
        torch.from_numpy(img_forward).permute(2, 0, 1),
        torch.from_numpy(img_backward).permute(2, 0, 1),
        mask_reconstructed.astype(np.float32),
        torch.from_numpy(overlay_lost).permute(2, 0, 1),
        mask_forward.astype(np.float32)
    ]
    
    make_grid(*imgs, titles=titles, out_path=out_path)
    
    # Print statistics
    total_pixels = mask_src_full.sum()
    reconstructed_pixels = mask_reconstructed.sum()
    lost_pixels = mask_lost.sum()
    
    print(f"\n=== Reconstruction Statistics ===")
    print(f"Original pixels: {total_pixels}")
    print(f"Reconstructed pixels: {reconstructed_pixels} ({100*reconstructed_pixels/total_pixels:.1f}%)")
    print(f"Lost pixels: {lost_pixels} ({100*lost_pixels/total_pixels:.1f}%)")
    
    return {
        "img_original": src_rgb,
        "img_forward": img_forward,
        "img_backward": img_backward,
        "mask_forward": mask_forward,
        "mask_backward": mask_backward,
        "mask_lost": mask_lost,
        "pts_world_all": pts_world,
        "pts_world_visible": pts_world_visible,
        "cols_all": cols,
        "cols_visible": cols_visible,
    }


def translate_camera(T_w2c, dx=0.0, dy=0.0, dz=0.0):
    """
    Translate camera in its local frame.
    dx: right(+)/left(-), dy: up(+)/down(-), dz: forward(+)/backward(-)
    """
    R, t = T_w2c[:3, :3], T_w2c[:3, 3]
    delta_cam = np.array([dx, dy, dz], dtype=np.float32)
    delta_world = R.T @ (-delta_cam)
    t_new = t + delta_world
    T_new = T_w2c.copy()
    T_new[:3, 3] = t_new
    return T_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Forward-backward warp data generation: "
                    "Warp image to new view, then warp back using only visible points"
    )
    parser.add_argument("--image", required=True, help="Path to RGB image")
    parser.add_argument("--depth", required=True, help="Path to depth map (.npy)")
    parser.add_argument("--K", required=True, help="Path to intrinsics matrix (.npy)")
    parser.add_argument("--T_src_w2c", required=True, help="Path to extrinsics (.npy)")
    parser.add_argument("--yaw", type=float, default=30.0, help="Yaw rotation in degrees")
    parser.add_argument("--dx", type=float, default=0.0, help="Translation right (+) / left (-) in meters")
    parser.add_argument("--dy", type=float, default=0.0, help="Translation up (+) / down (-) in meters")
    parser.add_argument("--dz", type=float, default=0.0, help="Translation forward (+) / backward (-) in meters")
    parser.add_argument("--out", default="out/forward_backward_warp.png", help="Output path")
    
    args = parser.parse_args()
    
    result = forward_backward_warp(
        args.image, args.depth, args.K, args.T_src_w2c,
        yaw_deg=args.yaw, dx=args.dx, dy=args.dy, dz=args.dz,
        out_path=args.out
    )

