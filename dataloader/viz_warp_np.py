import os
import math
import argparse
import numpy as np
import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt

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

def depth_to_points_cam(depth, K):
    """
    depth: (H,W) meters, K: (3,3)
    returns: (H*W,3) points in the *camera* frame and (H*W,2) pixel coords
    """
    H, W = depth.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    z = depth.reshape(-1)  # (HW,)
    x = xs.reshape(-1)
    y = ys.reshape(-1)

    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    X = (x - cx) / fx * z
    Y = (y - cy) / fy * z
    pts_cam = np.stack([X, Y, z], axis=-1)  # (HW,3)
    return pts_cam, np.stack([x, y], axis=-1)

def cam2world_from_world2cam(T_w2c):
    """Invert world→cam (OpenCV) to cam→world."""
    R = T_w2c[:3,:3]
    t = T_w2c[:3, 3]
    T_c2w = np.eye(4, dtype=np.float32)
    T_c2w[:3,:3] = R.T
    T_c2w[:3, 3] = -R.T @ t
    return T_c2w

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
    # np.unique keeps the first occurrence when not sorted; we need the first per pixel,
    # so we reverse-sort or use return_index with kind='stable'
    uniq, first = np.unique(idx, return_index=True)
    img[uniq] = cols[first]
    mask[uniq] = 1

    img  = img.reshape(h, w, 3)
    mask = mask.reshape(h, w)
    return img, mask

def overlay_mask(image01, mask, color=(1.0, 0.0, 0.0), alpha=0.35):
    """image01: HWC in [0,1], mask: HW (0/1)"""
    overlay = image01.copy()
    color_arr = np.array(color, dtype=np.float32).reshape(1,1,3)
    m = mask.astype(bool)
    overlay[m] = (1 - alpha) * overlay[m] + alpha * color_arr
    return np.clip(overlay, 0, 1)

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
        import open3d as o3d
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
        # Fallback: derive world points from depth + source RGB
        rgb = read_image(image_path).float() / 255.0  # (3,H,W)
        if rgb.shape[0] == 1: rgb = rgb.repeat(3,1,1)
        src_rgb = rgb.permute(1,2,0).numpy()  # HWC
        depth = np.load(depth_path).astype(np.float32)
        pts_cam, _ = depth_to_points_cam(depth, K)                 # (HW,3) cam frame
        T_src_c2w = cam2world_from_world2cam(T_src_w2c)
        pts_world = (T_src_c2w[:3,:3] @ pts_cam.T + T_src_c2w[:3,3:4]).T  # (HW,3)
        cols = src_rgb.reshape(-1, 3)
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
    parser = argparse.ArgumentParser("OpenCV world→cam warp visualizer (NumPy rasterizer)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--depth", default=None)
    parser.add_argument("--K", required=True)
    parser.add_argument("--T_src_w2c", required=True)
    parser.add_argument("--ply", default=None)
    parser.add_argument("--yaw_max", type=float, default=30.0, help="max yaw range (+/- degrees)")
    parser.add_argument("--frames", type=int, default=30, help="number of frames per half swing")
    parser.add_argument("--out_gif", default="out/warp_swing.gif", help="output GIF path")
    args = parser.parse_args()

    # Load shared data
    K = np.load(args.K).astype(np.float32)
    T_src_w2c = np.load(args.T_src_w2c).astype(np.float32)
    if T_src_w2c.shape == (3,4):
        E = np.eye(4, dtype=np.float32); E[:3,:4] = T_src_w2c; T_src_w2c = E

    # Build points & colors (reuse logic from visualize_one)
    if args.ply:
        import open3d as o3d
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
        pts_cam, _ = depth_to_points_cam(depth, K)
        T_src_c2w = cam2world_from_world2cam(T_src_w2c)
        pts_world = (T_src_c2w[:3,:3] @ pts_cam.T + T_src_c2w[:3,3:4]).T
        cols = src_rgb.reshape(-1, 3)

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

    visualize_side_by_side(args.image, args.depth, args.K, args.T_src_w2c, args.yaw_deg, args.out)
