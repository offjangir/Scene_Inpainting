import os
import math
import argparse
import numpy as np
import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
    NormWeightedCompositor  # Use this instead of AlphaCompositor
)
from pytorch3d.structures import Pointclouds
import pytorch3d.renderer.cameras

# ============================================================
# === Utilities ==============================================
# ============================================================
def make_grid(*imgs, titles=None, out_path=None, dpi=120):
    n = len(imgs)
    plt.figure(figsize=(4*n, 4), dpi=dpi)
    for i, im in enumerate(imgs, 1):
        plt.subplot(1, n, i)
        if im.ndim == 3 and im.shape[0] in (1, 3):   # CHW -> HWC torch
            im = im.permute(1, 2, 0).detach().cpu().clip(0, 1).numpy()
        elif im.ndim == 2:                            # HW
            im = np.clip(im, 0, 1)
        elif im.ndim == 3 and im.shape[-1] in (1, 3): # HWC
            im = np.clip(im, 0, 1)
        else:
            raise ValueError(f"Unexpected image shape {im.shape}")
        plt.imshow(im)
        plt.axis("off")
        if titles: plt.title(titles[i-1])
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()
    plt.close()

def depth_to_points(depth, rgb, K, extrinsic):
    """
    Use Open3D to unproject depth map directly to world points.
    depth: (H,W) in meters
    rgb: (H,W,3) in [0,1]
    extrinsic: (3,3) intrinsics
    T_w2c: (4,4) world→cam extrinsics
    Returns: pts_world (N,3), colors (N,3)
    """
    H, W = depth.shape
    
    # Create Open3D RGB-D image
    rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.float32))  # Open3D expects mm
    rgb_o3d = o3d.geometry.Image(rgb_uint8)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d, 
        depth_o3d, 
        depth_scale=1000.0, 
        convert_rgb_to_intensity=False
    )
    
    # Create Open3D camera intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    # Create point cloud directly in world coordinates
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic)
    pts_world = np.asarray(pcd.points).astype(np.float32)
    colors = np.asarray(pcd.colors).astype(np.float32)
    return pts_world, colors


def apply_cam_yaw(T_w2c, degrees):
    """
    Rotate camera in-place by yaw (+deg to the right) in *camera* coords.
    Returns a new world→cam extrinsic.
    """
    R, t = T_w2c[:3,:3], T_w2c[:3,3]
    th = np.deg2rad(degrees)
    R_delta = np.array([[ np.cos(th), 0,  np.sin(th)],
                        [ 0,          1,  0         ],
                        [-np.sin(th), 0,  np.cos(th)]], dtype=np.float32)
    R_tgt = R @ R_delta
    T_tgt = np.eye(4, dtype=np.float32)
    T_tgt[:3,:3] = R_tgt
    T_tgt[:3, 3] = t
    return T_tgt


def apply_cam_transform(T_w2c, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, 
                       translate_x=0.0, translate_y=0.0, translate_z=0.0):
    """
    Apply full 6-DOF transformation to camera in camera coordinates.
    
    Args:
        T_w2c: (4,4) world→cam extrinsic matrix
        yaw_deg: rotation around Y axis (left/right)
        pitch_deg: rotation around X axis (up/down)
        roll_deg: rotation around Z axis (tilt)
        translate_x: translation along camera X axis (right)
        translate_y: translation along camera Y axis (down)
        translate_z: translation along camera Z axis (forward)
    
    Returns:
        T_tgt: (4,4) transformed world→cam extrinsic
    """
    R, t = T_w2c[:3, :3], T_w2c[:3, 3]
    
    # Convert angles to radians
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    
    # Rotation matrices in camera coordinates
    # Yaw (Y-axis rotation)
    R_yaw = np.array([
        [np.cos(yaw),  0, np.sin(yaw)],
        [0,            1, 0          ],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ], dtype=np.float32)
    
    # Pitch (X-axis rotation)
    R_pitch = np.array([
        [1, 0,              0             ],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ], dtype=np.float32)
    
    # Roll (Z-axis rotation)
    R_roll = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll),  np.cos(roll), 0],
        [0,             0,            1]
    ], dtype=np.float32)
    
    # Combined rotation: apply roll, then pitch, then yaw
    R_delta = R_yaw @ R_pitch @ R_roll
    R_tgt = R @ R_delta
    
    # Translation in camera coordinates
    t_delta = np.array([translate_x, translate_y, translate_z], dtype=np.float32)
    # Transform translation to world coordinates and apply
    t_tgt = t + R @ t_delta
    
    # Construct output transformation
    T_tgt = np.eye(4, dtype=np.float32)
    T_tgt[:3, :3] = R_tgt
    T_tgt[:3, 3] = t_tgt
    
    return T_tgt


def rasterize_points_world(pts_world, colors01, K, T_w2c, w, h, z_clip=1e-4):
    """
    Vectorized z-buffer rasterization from world points.
    pts_world: (N,3), colors01: (N,3) in [0,1]
    K: (3,3), T_w2c: (4,4) world→cam (OpenCV)
    Returns: img(H,W,3) in [0,1], mask(H,W) in {0,1}
    """
    # world -> cam
    pts_h = np.concatenate([pts_world, np.ones((pts_world.shape[0],1), dtype=np.float32)], axis=1)  # (N,4)
    cam = (T_w2c @ pts_h.T).T  # (N,4)
    z = cam[:,2]
    keep = z > z_clip
    if not np.any(keep):
        return np.zeros((h,w,3), np.float32), np.zeros((h,w), np.uint8)
    cam = cam[keep]
    z   = z[keep]
    cols = colors01[keep]

    # project
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    u = fx * (cam[:,0] / z) + cx
    v = fy * (cam[:,1] / z) + cy
    
    # in-bounds
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(valid):
        return np.zeros((h,w,3), np.float32), np.zeros((h,w), np.uint8)

    u = u[valid].astype(np.int32)
    v = v[valid].astype(np.int32)
    z = z[valid]
    cols = cols[valid]

    # depth test (closest wins) — vectorized
    idx = v * w + u
    order = np.argsort(z)         # front to back
    idx = idx[order]
    cols = cols[order]

    img  = np.zeros((h*w, 3), np.float32)
    mask = np.zeros(h*w, np.uint8)

    # keep first hit per pixel (closest due to sorting)
    uniq, first = np.unique(idx, return_index=True)
    img[uniq] = cols[first]
    mask[uniq] = 1

    img  = img.reshape(h, w, 3)
    mask = mask.reshape(h, w)
    return img, mask

def project_invalid(pts_world, colors01, K, T_w2c, w, h, z_clip=1e-4):
    """
    PyTorch3D version that returns VISIBLE mask
    Returns: boolean mask where True = point is VISIBLE in the target view
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    N = pts_world.shape[0]
    
    # Step 1: Transform to camera space
    pts_h = np.concatenate([pts_world, np.ones((N, 1), dtype=np.float32)], axis=1)
    cam = (T_w2c @ pts_h.T).T
    z = cam[:, 2]
    
    # Step 2: Filter points behind camera
    behind_camera = z <= z_clip
    
    # Step 3: Project to image space
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (cam[:, 0] / z) + cx
    v = fy * (cam[:, 1] / z) + cy
    
    # Step 4: Check out of bounds
    out_of_bounds = (u < 0) | (u >= w) | (v >= h) | (v < 0)
    
    # Step 5: Filter valid points for PyTorch3D rendering
    valid = ~behind_camera & ~out_of_bounds
    
    if not np.any(valid):
        # All points invisible
        return np.zeros(N, dtype=bool)  # Changed to zeros for VISIBLE mask
    
    # ===== KEY FIX: Use camera-space coordinates =====
    pts_cam_valid = cam[valid, :3]  # Already in camera space!
    
    # Convert to torch
    pts_valid = torch.from_numpy(pts_cam_valid).float().to(device)[None]
    cols_valid = torch.from_numpy(colors01[valid]).float().to(device)[None]
    
    # Map from valid indices back to original indices
    valid_indices = np.where(valid)[0]
    
    # Create point cloud
    point_cloud = Pointclouds(points=pts_valid, features=cols_valid)
    
    # Setup rasterizer with splatting
    raster_settings = PointsRasterizationSettings(
        image_size=(h, w),
        radius=0.01,  # Adjust for point density
        points_per_pixel=4,
        bin_size=None
    )
    
    # ===== KEY FIX: Identity camera since points are already in camera space =====
    cameras = pytorch3d.renderer.cameras.PerspectiveCameras(
        focal_length=torch.tensor([[fx, fy]], device=device, dtype=torch.float32),
        principal_point=torch.tensor([[cx, cy]], device=device, dtype=torch.float32),
        image_size=torch.tensor([[h, w]], device=device),
        device=device,
        in_ndc=False  # IMPORTANT! Use screen coordinates
    )
    
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    
    # Get fragments
    fragments = rasterizer(point_cloud)
    
    # fragments.idx shape: (1, h, w, points_per_pixel)
    # Get which valid points are visible (referenced in any pixel)
    visible_point_ids = fragments.idx[0].cpu().numpy()  # (h, w, points_per_pixel)
    visible_point_ids = visible_point_ids[visible_point_ids >= 0]  # Filter out -1 (no point)
    visible_point_ids_unique = np.unique(visible_point_ids)
    
    # Map back from valid subset to original indices
    visible_original_indices = valid_indices[visible_point_ids_unique]
    
    # ===== FIXED LOGIC: Create VISIBLE mask (True = visible) =====
    visible_mask = np.zeros(N, dtype=bool)  # Start with all False (invisible)
    visible_mask[visible_original_indices] = True  # Mark visible points as True
    
    # Points behind camera or out of bounds remain False (invisible)
    
    # Debug: Render image
    from pytorch3d.renderer import PointsRenderer, NormWeightedCompositor
    
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=NormWeightedCompositor(background_color=(0.0, 0.0, 0.0))
    )
    
    images = renderer(point_cloud)  # (1, h, w, 3)
    img = images[0].cpu().numpy()
    
    # Save debug image
    cv2.imwrite("debug_pytorch3d.png", (img[..., ::-1] * 255).astype(np.uint8))
    
    return visible_mask

def project_invalid_(pts_world, colors01, K, T_w2c, w, h, z_clip=1e-4):
    N = pts_world.shape[0]
    original_indices = np.arange(N)
    
    # world -> cam
    pts_h = np.concatenate([pts_world, np.ones((N, 1), dtype=np.float32)], axis=1)
    cam = (T_w2c @ pts_h.T).T
    z = cam[:, 2]
    
    # Track which points survive each filter
    keep = z > z_clip
    cam = cam[keep]
    z = z[keep]
    cols = colors01[keep]
    indices = original_indices[keep]
    
    if not np.any(keep):
        return (
            np.zeros((h, w, 3), np.float32),
            np.zeros((h, w), np.uint8),
            pts_world.copy(),
            colors01.copy(),
        )
    
    # project
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * (cam[:, 0] / z) + cx
    v = fy * (cam[:, 1] / z) + cy
    
    # in-bounds
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[valid].astype(np.int32)
    v = v[valid].astype(np.int32)
    z = z[valid]
    cols = cols[valid]
    indices = indices[valid]
    
    if len(u) == 0:
        return (
            np.zeros((h, w, 3), np.float32),
            np.zeros((h, w), np.uint8),
            pts_world.copy(),
            colors01.copy(),
        )
    
    # depth test
    idx = v * w + u
    order = np.argsort(z)
    idx = idx[order]
    cols = cols[order]
    indices = indices[order]
    
    img = np.zeros((h * w, 3), np.float32)
    mask = np.zeros(h * w, np.uint8)
    
    uniq, first = np.unique(idx, return_index=True)
    img[uniq] = cols[first]
    mask[uniq] = 1
    
    img = img.reshape(h, w, 3)
    mask = mask.reshape(h, w)
    
    # Mark which original points are visible
    visible_indices = indices[first]
    invisible_mask = np.ones(N, dtype=bool)
    invisible_mask[visible_indices] = False
    
    invisible_pts_world = pts_world[invisible_mask]
    invisible_cols = colors01[invisible_mask]
    # invisible_mask = valid
    # breakpoint()
    return invisible_mask

def overlay_mask(image01, mask, color=(1.0, 0.0, 0.0), alpha=0.35):
    """image01: HWC in [0,1], mask: HW (0/1)"""
    overlay = image01.copy()
    color_arr = np.array(color, dtype=np.float32).reshape(1,1,3)
    m = mask.astype(bool)
    overlay[m] = (1 - alpha) * overlay[m] + alpha * color_arr
    return np.clip(overlay, 0, 1)

def generate_warped_images(
    image_path, depth_path, K_path, T_src_w2c_path, T_tgt_w2c_path=None,
    ply_path=None, yaw_deg=30.0
):
    """
    Generate forward and backward warped images.
    
    Args:
        image_path: Path to source RGB image
        depth_path: Path to depth map (.npy)
        K_path: Path to camera intrinsics (.npy)
        T_src_w2c_path: Path to source world-to-camera transform (.npy)
        T_tgt_w2c_path: Path to target world-to-camera transform (.npy), optional
        ply_path: Path to point cloud (.ply), optional (uses this instead of depth if provided)
        yaw_deg: Yaw angle in degrees for target view (if T_tgt_w2c_path not provided)
    
    Returns:
        src_rgb: Source RGB image (H, W, 3) in [0, 1]
        img_fwd: Forward warped image at source pose (H, W, 3) in [0, 1]
        img_bwd: Backward warped image at target pose (H, W, 3) in [0, 1]
        mask_fwd: Mask for forward warp (H, W) binary
        mask_bwd: Mask for backward warp (H, W) binary
        pts_world: World points (N, 3)
        colors: Point colors (N, 3)
    """
    # Load intrinsics
    K = np.load(K_path).astype(np.float32)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    H = int(cy*2) if cy*2 > 0 else 512
    W = int(cx*2) if cx*2 > 0 else 512
    
    # Load extrinsics
    T_src_w2c = np.load(T_src_w2c_path).astype(np.float32)
    if T_src_w2c.shape == (3,4):
        E = np.eye(4, dtype=np.float32)
        E[:3,:4] = T_src_w2c
        T_src_w2c = E
        
    if T_tgt_w2c_path is not None:
        T_tgt_w2c = np.load(T_tgt_w2c_path).astype(np.float32)
        if T_tgt_w2c.shape == (3,4):
            E = np.eye(4, dtype=np.float32)
            E[:3,:4] = T_tgt_w2c
            T_tgt_w2c = E
    else:
        T_tgt_w2c = apply_cam_yaw(T_src_w2c, yaw_deg)
    
    # Build world points
    if ply_path is not None:
        pcd = o3d.io.read_point_cloud(ply_path)
        if len(pcd.colors) == 0:
            pcd.paint_uniform_color([1,1,1])
        pts_world = np.asarray(pcd.points).astype(np.float32)
        cols = np.asarray(pcd.colors).astype(np.float32)
        if cols.max() > 1.0:
            cols = cols / 255.0
        src_rgb, _ = rasterize_points_world(pts_world, cols, K, T_src_w2c, W, H)
    else:
        rgb = read_image(image_path).float() / 255.0
        if rgb.shape[0] == 1:
            rgb = rgb.repeat(3,1,1)
        src_rgb = rgb.permute(1,2,0).numpy()
        depth = np.load(depth_path).astype(np.float32)
        pts_world, cols = depth_to_points(depth, src_rgb, K, T_src_w2c)
    
    # Render forward and backward
    img_fwd, mask_fwd = rasterize_points_world(pts_world, cols, K, T_src_w2c, W, H)
    img_bwd, mask_bwd = rasterize_points_world(pts_world, cols, K, T_tgt_w2c, W, H)
    
    return src_rgb, img_fwd, img_bwd, mask_fwd, mask_bwd, pts_world, cols

# ============================================================
# === Core visualization =====================================
# ============================================================
def visualize_one(
    image_path, depth_path, K_path=None, T_src_w2c_path=None, T_tgt_w2c_path=None,
    ply_path=None, yaw_deg=30.0,
    out_path="out/vis.png"
):
    # Load intrinsics & image size
    if K_path is not None:
        K = np.load(K_path).astype(np.float32)
    else:
        # Infer K from image size if missing (fallback)
        rgb = read_image(image_path).float() / 255.0
        H, W = rgb.shape[-2], rgb.shape[-1]
        K = np.array([[525., 0.,   W/2.0],
                      [0.,   525., H/2.0],
                      [0.,   0.,   1.  ]], dtype=np.float32)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    H = int(cy*2) if cy*2 > 0 else 512
    W = int(cx*2) if cx*2 > 0 else 512

    # Load extrinsics (world→cam)
    if T_src_w2c_path is None:
        raise ValueError("Please provide --T_src_w2c (world→cam, OpenCV).")
    T_src_w2c = np.load(T_src_w2c_path).astype(np.float32)
    if T_src_w2c.shape == (3,4):
        E = np.eye(4, dtype=np.float32); E[:3,:4] = T_src_w2c; T_src_w2c = E
    if T_tgt_w2c_path is not None:
        T_tgt_w2c = np.load(T_tgt_w2c_path).astype(np.float32)
        if T_tgt_w2c.shape == (3,4):
            E = np.eye(4, dtype=np.float32); E[:3,:4] = T_tgt_w2c; T_tgt_w2c = E
    else:
        T_tgt_w2c = apply_cam_yaw(T_src_w2c, yaw_deg)  # in-place yaw

    # Build world points + per-point colors in [0,1]
    if ply_path is not None:
        pcd = o3d.io.read_point_cloud(ply_path)
        if len(pcd.colors) == 0:
            pcd.paint_uniform_color([1,1,1])
        pts_world = np.asarray(pcd.points).astype(np.float32)
        cols = np.asarray(pcd.colors).astype(np.float32)
        if cols.max() > 1.0:
            cols = cols / 255.0
        # Render forward & backward directly from world points:
        img_fwd, mask_fwd = rasterize_points_world(pts_world, cols, K, T_src_w2c, W, H)
        img_bwd, mask_bwd = rasterize_points_world(pts_world, cols, K, T_tgt_w2c, W, H)
        src_rgb = img_fwd.copy()  # best available approximation of source RGB from the cloud
    else:
        # Use Open3D unprojection from depth + source RGB
        rgb = read_image(image_path).float() / 255.0  # (3,H,W)
        if rgb.shape[0] == 1: rgb = rgb.repeat(3,1,1)
        src_rgb = rgb.permute(1,2,0).numpy()  # HWC
        depth = np.load(depth_path).astype(np.float32)
        
        # Use Open3D to get world points directly
        pts_world, cols = depth_to_points(depth, src_rgb, K, T_src_w2c)
        
        img_fwd, mask_fwd = rasterize_points_world(pts_world, cols, K, T_src_w2c, W, H)
        img_bwd, mask_bwd = rasterize_points_world(pts_world, cols, K, T_tgt_w2c, W, H)

    # Consistency
    mask_consistent = (mask_fwd > 0) & (mask_bwd > 0)
    mask_cond = (1 - mask_consistent.astype(np.float32))

    # Overlays
    overlay_fwd = overlay_mask(img_fwd, mask_consistent, color=(0,1,0))   # green on fwd
    overlay_bwd = overlay_mask(img_bwd, mask_consistent, color=(1,0,0))   # red on bwd
    hole_overlay = overlay_mask(src_rgb, (mask_cond > 0.5).astype(np.uint8), color=(1,0,0))

    titles = [
        "Source RGB (approx)",
        "Fwd render (src pose)",
        "Bwd render (tgt pose)",
        "Valid mask (∩)",
        "Masked src (holes in red)",
        "Hole mask"
    ]
    imgs = [
        torch.from_numpy(src_rgb).permute(2,0,1),                  # CHW
        torch.from_numpy(img_fwd).permute(2,0,1),
        torch.from_numpy(img_bwd).permute(2,0,1),
        mask_consistent.astype(np.float32),
        torch.from_numpy(hole_overlay).permute(2,0,1),
        mask_cond.astype(np.float32)
    ]
    make_grid(*imgs, titles=titles, out_path=out_path)

# ============================================================
# === CLI ====================================================
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("OpenCV world→cam warp visualizer with Open3D unprojection")
    parser.add_argument("--image", required=True)
    parser.add_argument("--depth", default=None)
    parser.add_argument("--K", required=True)
    parser.add_argument("--T_src_w2c", required=True)
    parser.add_argument("--ply", default=None)
    parser.add_argument("--yaw_deg", type=float, default=30.0, help="yaw angle in degrees")
    parser.add_argument("--yaw_max", type=float, default=30.0, help="max yaw range (+/- degrees) for GIF")
    parser.add_argument("--frames", type=int, default=30, help="number of frames per half swing")
    parser.add_argument("--out", default="out/vis.png", help="output visualization path")
    parser.add_argument("--out_gif", default="out/warp_swing.gif", help="output GIF path")
    parser.add_argument("--make_gif", action="store_true", help="Generate swinging GIF")
    args = parser.parse_args()

    if args.make_gif:
        # Load shared data
        K = np.load(args.K).astype(np.float32)
        T_src_w2c = np.load(args.T_src_w2c).astype(np.float32)
        if T_src_w2c.shape == (3,4):
            E = np.eye(4, dtype=np.float32); E[:3,:4] = T_src_w2c; T_src_w2c = E

        # Build points & colors using Open3D
        if args.ply:
            pcd = o3d.io.read_point_cloud(args.ply)
            if len(pcd.colors) == 0:
                pcd.paint_uniform_color([1,1,1])
            pts_world = np.asarray(pcd.points).astype(np.float32)
            cols = np.asarray(pcd.colors).astype(np.float32)
            if cols.max() > 1.0: cols /= 255.0
        else:
            rgb = read_image(args.image).float() / 255.0
            depth = np.load(args.depth).astype(np.float32)
            if rgb.shape[0] == 1: rgb = rgb.repeat(3,1,1)
            src_rgb = rgb.permute(1,2,0).numpy()
            
            # Use Open3D to get world points directly
            pts_world, cols = depth_to_points(depth, src_rgb, K, T_src_w2c)

        # Prepare swing path (yaw -max → +max → -max)
        yaws = np.linspace(-args.yaw_max, args.yaw_max, args.frames)
        yaws = np.concatenate([yaws, yaws[::-1]])  # back and forth
        H, W = int(K[1,2]*2), int(K[0,2]*2)

        frames = []
        for deg in yaws:
            T_tgt = apply_cam_yaw(T_src_w2c, deg)
            img, _ = rasterize_points_world(pts_world, cols, K, T_tgt, W, H)
            frames.append((np.clip(img * 255, 0, 255)).astype(np.uint8))

        import imageio
        os.makedirs(os.path.dirname(args.out_gif), exist_ok=True)
        imageio.mimsave(args.out_gif, frames, fps=15)
        print(f"Saved swinging GIF to {args.out_gif}")
    else:
        visualize_one(args.image, args.depth, args.K, args.T_src_w2c, 
                     yaw_deg=args.yaw_deg, ply_path=args.ply, out_path=args.out)
