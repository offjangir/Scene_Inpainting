"""
Shared dataset utilities for inpainting training.
"""
import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class InpaintPairDataset(Dataset):
    """
    Dataset for ControlNet-based inpainting training.
    
    Expected directory structure:
        root/
            scene1/
                pair1/
                    image_target.png    # Ground truth complete image
                    image_cond.png      # Masked/incomplete image
                    mask_cond.png       # Binary mask (1=hole, 0=valid)
                pair2/
                    ...
            scene2/
                ...
    
    Mask convention (controlled by invert_mask parameter):
    - If invert_mask=False: 1 = holes to inpaint, 0 = valid regions (default)
    - If invert_mask=True:  0 = holes to inpaint, 1 = valid regions (inverted)
    """
    
    def __init__(
        self,
        root,
        size=512,
        image_suffix="_target.png",
        cond_suffix="_cond.png",
        mask_suffix="_mask.png",
        invert_mask=False
    ):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory containing scene/pair structure
            size: Target image size for resizing
            image_suffix: Suffix for target images (default: "_target.png")
            cond_suffix: Suffix for conditioning images (default: "_cond.png")
            mask_suffix: Suffix for mask images (default: "_mask.png")
            invert_mask: Whether to invert mask convention
        """
        self.root = root
        self.size = size
        self.invert_mask = invert_mask
        self.samples = []
        
        if not os.path.exists(root):
            raise ValueError(f"Dataset root directory does not exist: {root}")
        
        # Traverse all scene folders
        for scene_dir in sorted(os.listdir(root)):
            scene_path = os.path.join(root, scene_dir)
            if not os.path.isdir(scene_path):
                continue
            
            # Traverse all pair folders inside each scene
            for pair_dir in sorted(os.listdir(scene_path)):
                pair_path = os.path.join(scene_path, pair_dir)
                if not os.path.isdir(pair_path):
                    continue

                # Construct file paths
                img_path = os.path.join(pair_path, f"image{image_suffix}")
                cond_path = os.path.join(pair_path, f"image{cond_suffix}")
                mask_path = os.path.join(pair_path, f"mask{mask_suffix}")

                # Only add valid samples
                if all(os.path.exists(p) for p in [img_path, cond_path, mask_path]):
                    self.samples.append({
                        "image": img_path,
                        "conditioning": cond_path,
                        "mask": mask_path
                    })
                else:
                    print(f"[⚠️] Skipping incomplete pair: {pair_path}")

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {root}")
        
        print(f"✅ Found {len(self.samples)} valid samples from {root}")

        # Define transforms
        self.tform_rgb = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        self.tform_mask = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            dict with keys:
                - image: Target image tensor [3, H, W]
                - conditioning_image: Conditioning image tensor [3, H, W]
                - mask: Mask tensor [1, H, W] (1=hole, 0=valid after processing)
        """
        s = self.samples[idx]
        
        try:
            img = self.tform_rgb(Image.open(s["image"]).convert("RGB"))
            cond = self.tform_rgb(Image.open(s["conditioning"]).convert("RGB"))
            mask = self.tform_mask(Image.open(s["mask"]).convert("L"))
        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx} from {s}: {e}")
        
        # Invert mask if needed (swap 0s and 1s)
        if self.invert_mask:
            mask = 1.0 - mask
        
        return {
            "image": img,
            "conditioning_image": cond,
            "mask": mask  # Now guaranteed: 1 = hole to inpaint, 0 = valid
        }

