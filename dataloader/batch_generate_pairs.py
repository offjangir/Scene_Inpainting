"""
Batch generate training pairs from a multi-view dataset.
Assumes directory structure:
    dataset/
        scene85/
            images0/
                0.jpg
                1.jpg
                ...
            depth/
                0.npy
                1.npy
                ...
            intrinsics.npy
            extrinsics.npy
        scene86/
            ...
            
This script uses only 0.jpg and 0.npy from each scene.
"""

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from generate_warp_pairs import generate_warp_pair


def batch_generate(
    input_dir,
    output_dir,
    yaw_range=(-30, 30),
    num_samples_per_scene=5,
    dx_range=(-0.5, 0.5),
    dy_range=(-0.2, 0.2),
    dz_range=(0.0, 0.0),
    seed=42
):
    """
    Generate training pairs for all scenes in input directory.
    
    Args:
        input_dir: Root directory containing scene subdirectories
        output_dir: Output directory for generated pairs
        yaw_range: (min, max) yaw rotation in degrees
        num_samples_per_scene: Number of warped views to generate per scene
        dx_range: (min, max) horizontal translation in meters
        dy_range: (min, max) vertical translation in meters
        dz_range: (min, max) depth translation in meters
        seed: Random seed for reproducibility
    """
    
    np.random.seed(seed)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all scenes (directories with required files)
    scenes = []
    for scene_dir in sorted(input_path.iterdir()):
        if not scene_dir.is_dir():
            continue
        
        # Check for required files (using 0.jpg and 0.npy)
        image_file = scene_dir / "images0" / "0.jpg"
        depth_file = scene_dir / "depth" / "0.npy"
        K_file = scene_dir / "intrinsics.npy"
        T_file = scene_dir / "extrinsics.npy"
        
        if all([f.exists() for f in [image_file, depth_file, K_file, T_file]]):
            scenes.append(scene_dir)
        else:
            print(f"⚠️ Skipping {scene_dir.name}: missing required files")
    
    print(f"Found {len(scenes)} scenes in {input_dir}")
    print(f"Generating {num_samples_per_scene} samples per scene...")
    print(f"Total pairs to generate: {len(scenes) * num_samples_per_scene}")
    print(f"\nParameters:")
    print(f"  Yaw range: {yaw_range}°")
    print(f"  dx range: {dx_range}m")
    print(f"  dy range: {dy_range}m")
    print(f"  dz range: {dz_range}m")
    print()
    
    # Generate pairs
    metadata = []
    sample_idx = 0
    
    for scene_dir in tqdm(scenes, desc="Processing scenes"):
        scene_name = scene_dir.name
        
        # File paths (using 0.jpg and 0.npy)
        image_file = str(scene_dir / "images0" / "0.jpg")
        depth_file = str(scene_dir / "depth" / "0.npy")
        K_file = str(scene_dir / "intrinsics.npy")
        T_file = str(scene_dir / "extrinsics.npy")
        
        # Generate multiple warped views for this scene
        for i in range(num_samples_per_scene):
            # Random camera parameters
            yaw = np.random.uniform(*yaw_range)
            dx = np.random.uniform(*dx_range)
            dy = np.random.uniform(*dy_range)
            dz = np.random.uniform(*dz_range)
            
            # Create output directory: output_dir/scene_name/pair_i/
            pair_dir = output_path / scene_name / f"pair_{i}"
            pair_dir.mkdir(parents=True, exist_ok=True)
            
            # Use simple basenames (files will be in their own directories)
            basename = "sample"
            
            try:
                result = generate_warp_pair(
                    image_file, depth_file, K_file, T_file,
                    yaw_deg=yaw, dx=dx, dy=dy, dz=dz,
                    out_dir=str(pair_dir),
                    basename=basename
                )
                
                # Store metadata
                metadata.append({
                    "sample_id": sample_idx,
                    "scene": scene_name,
                    "pair_id": i,
                    "source_view": "0",
                    "pair_dir": str(pair_dir.relative_to(output_path)),
                    "yaw": float(yaw),
                    "dx": float(dx),
                    "dy": float(dy),
                    "dz": float(dz),
                    "target_path": str(Path(result["target_path"]).relative_to(output_path)),
                    "cond_path": str(Path(result["cond_path"]).relative_to(output_path)),
                    "mask_path": str(Path(result["mask_path"]).relative_to(output_path)),
                    "stats": result["stats"]
                })
                
                sample_idx += 1
                
            except Exception as e:
                print(f"\n⚠️ Error processing {scene_name} sample {i}: {e}")
                continue
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Generated {len(metadata)} training pairs")
    print(f"📁 Output directory: {output_path}")
    print(f"📄 Metadata saved to: {metadata_path}")
    
    # Print statistics
    if metadata:
        hole_percentages = [m["stats"]["hole_percentage"] for m in metadata]
        print(f"\n=== Dataset Statistics ===")
        print(f"Average hole percentage: {np.mean(hole_percentages):.1f}%")
        print(f"Min hole percentage: {np.min(hole_percentages):.1f}%")
        print(f"Max hole percentage: {np.max(hole_percentages):.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch generate training pairs from a dataset"
    )
    parser.add_argument("--input_dir", required=True, help="Input directory with scene subdirectories")
    parser.add_argument("--output_dir", required=True, help="Output directory for training pairs")
    parser.add_argument("--yaw_min", type=float, default=-30.0, help="Minimum yaw rotation (degrees)")
    parser.add_argument("--yaw_max", type=float, default=30.0, help="Maximum yaw rotation (degrees)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per scene")
    parser.add_argument("--dx_min", type=float, default=-0.5, help="Minimum horizontal translation (m)")
    parser.add_argument("--dx_max", type=float, default=0.5, help="Maximum horizontal translation (m)")
    parser.add_argument("--dy_min", type=float, default=-0.2, help="Minimum vertical translation (m)")
    parser.add_argument("--dy_max", type=float, default=0.2, help="Maximum vertical translation (m)")
    parser.add_argument("--dz_min", type=float, default=0.0, help="Minimum depth translation (m)")
    parser.add_argument("--dz_max", type=float, default=0.0, help="Maximum depth translation (m)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    batch_generate(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        yaw_range=(args.yaw_min, args.yaw_max),
        num_samples_per_scene=args.num_samples,
        dx_range=(args.dx_min, args.dx_max),
        dy_range=(args.dy_min, args.dy_max),
        dz_range=(args.dz_min, args.dz_max),
        seed=args.seed
    )

