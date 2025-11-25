import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

# Reuse the utility functions you already defined:
from viz_warp_np import (
    depth_to_points_cam, cam2world_from_world2cam,
    apply_cam_yaw, rasterize_points_world
)

class WarpInpaintDataset(Dataset):
    """
    Generates (image_cond, mask_cond, image_target, T_src, T_tgt)
    triplets for inpainting supervision.
    Supports RGB+Depth input with OpenCV-style intrinsics/extrinsics.
    """

    def __init__(self, root_dir, yaw_range=(-15, 15), move_x_range=(-0.1, 0.1), transform=None):
        """
        Args:
            root_dir: scene directory containing:
                images0/, depth/, intrinsics.npy, extrinsics.npy
            yaw_range: degrees for random yaw rotation
            move_x_range: meters for random left-right translation
            transform: optional callable for augmentation
        """
        self.root = root_dir
        self.image_dir = os.path.join(root_dir, "images0")
        self.depth_dir = os.path.join(root_dir, "depth")
        self.K_path = os.path.join(root_dir, "intrinsics.npy")
        self.T_path = os.path.join(root_dir, "extrinsics.npy")
        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith((".png", ".jpg"))]
        )
        self.yaw_range = yaw_range
        self.move_x_range = move_x_range
        self.transform = transform

        self.K = np.load(self.K_path).astype(np.float32)
        self.extr = np.load(self.T_path).astype(np.float32)
        if self.extr.ndim == 2:
            self.extr = self.extr[None, ...]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # ---- Load RGB + Depth ----
        fname = self.image_files[idx]
        img_path = os.path.join(self.image_dir, fname)
        dep_path = os.path.join(self.depth_dir, fname.replace(".jpg", ".npy").replace(".png", ".npy"))

        rgb = read_image(img_path).float() / 255.0   # (3,H,W)
        depth = torch.from_numpy(np.load(dep_path).astype(np.float32))  # (H,W)
        if rgb.shape[0] == 1: rgb = rgb.repeat(3,1,1)
        H, W = rgb.shape[-2], rgb.shape[-1]

        K = self.K
        T_src_w2c = self.extr[min(idx, len(self.extr)-1)]
        if T_src_w2c.shape == (3,4):
            E = np.eye(4, dtype=np.float32); E[:3,:4] = T_src_w2c; T_src_w2c = E

        # ---- Build world points ----
        pts_cam, _ = depth_to_points_cam(depth.numpy(), K)
        T_src_c2w = cam2world_from_world2cam(T_src_w2c)
        pts_world = (T_src_c2w[:3,:3] @ pts_cam.T + T_src_c2w[:3,3:4]).T
        cols = rgb.permute(1,2,0).numpy().reshape(-1,3)

        # ---- Random pose perturbation ----
        yaw = np.random.uniform(*self.yaw_range)
        dx  = np.random.uniform(*self.move_x_range)
        T_tgt_w2c = translate_camera(apply_cam_yaw(T_src_w2c, yaw), dx=dx)

        # ---- Rasterize forward/backward ----
        img_fwd, mask_fwd = rasterize_points_world(pts_world, cols, K, T_src_w2c, W, H)
        img_bwd, mask_bwd = rasterize_points_world(pts_world, cols, K, T_tgt_w2c, W, H)

        mask_valid = (mask_fwd > 0) & (mask_bwd > 0)
        mask_cond = (1 - mask_valid.astype(np.float32))
        image_cond = rgb.numpy().transpose(1,2,0) * mask_valid[...,None]
        image_target = rgb.numpy().transpose(1,2,0)

        # ---- Convert to tensors ----
        sample = {
            "image_cond": torch.from_numpy(image_cond).permute(2,0,1),
            "mask_cond": torch.from_numpy(mask_cond).unsqueeze(0),
            "image_target": torch.from_numpy(image_target).permute(2,0,1),
            "T_src": torch.from_numpy(T_src_w2c),
            "T_tgt": torch.from_numpy(T_tgt_w2c),
        }

        if self.transform:
            sample = self.transform(sample)
        return sample


def translate_camera(T_w2c, dx=0.0, dy=0.0, dz=0.0):
    """
    Translate camera in its local frame.
    dx: right(+)/left(-), dy: up(+)/down(-), dz: forward(+)/backward(-)
    """
    R, t = T_w2c[:3,:3], T_w2c[:3,3]
    delta_cam = np.array([dx, dy, dz], dtype=np.float32)
    delta_world = R.T @ (-delta_cam)
    t_new = t + delta_world
    T_new = T_w2c.copy()
    T_new[:3,3] = t_new
    return T_new
