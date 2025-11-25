"""
Script to log inference results to Weights & Biases (wandb)

This script scans a results directory and logs all inference images to wandb
for easy visualization and comparison.

Usage:
    # Log all results from a directory
    python log_results_to_wandb.py \
        --results_dir results2 \
        --wandb_project "inpainting-results" \
        --wandb_run_name "inference-run-1"

    # Log only comparison images
    python log_results_to_wandb.py \
        --results_dir results2 \
        --image_type comparison \
        --wandb_project "inpainting-results"

    # Log specific scenes only
    python log_results_to_wandb.py \
        --results_dir results2 \
        --scenes scene85 scene86 \
        --wandb_project "inpainting-results"

    # Log as a wandb table (better for browsing)
    python log_results_to_wandb.py \
        --results_dir results2 \
        --log_as_table \
        --wandb_project "inpainting-results"
"""

import os
import argparse
import wandb
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import glob


def find_result_images(results_dir, image_type="both", scenes=None):
    """
    Find all result images in the results directory.
    
    Args:
        results_dir: Root directory containing scene/pair subdirectories
        image_type: "both", "comparison", or "inpainted"
        scenes: List of scene names to include (None = all)
    
    Returns:
        List of dicts with image info: {scene, pair, path, type}
    """
    results = []
    
    # Find all scene directories
    scene_dirs = sorted([d for d in os.listdir(results_dir) 
                        if os.path.isdir(os.path.join(results_dir, d))])
    
    if scenes:
        scene_dirs = [s for s in scene_dirs if s in scenes]
    
    for scene in scene_dirs:
        scene_path = os.path.join(results_dir, scene)
        
        # Find all pair directories
        pair_dirs = sorted([d for d in os.listdir(scene_path)
                           if os.path.isdir(os.path.join(scene_path, d))])
        
        for pair in pair_dirs:
            pair_path = os.path.join(scene_path, pair)
            
            # Look for images
            comparison_path = os.path.join(pair_path, "inpainted_comparison.png")
            inpainted_path = os.path.join(pair_path, "inpainted.png")
            
            if image_type in ["both", "comparison"] and os.path.exists(comparison_path):
                results.append({
                    "scene": scene,
                    "pair": pair,
                    "path": comparison_path,
                    "type": "comparison"
                })
            
            if image_type in ["both", "inpainted"] and os.path.exists(inpainted_path):
                results.append({
                    "scene": scene,
                    "pair": pair,
                    "path": inpainted_path,
                    "type": "inpainted"
                })
    
    return results


def log_images_as_gallery(results, wandb_project, wandb_run_name=None, group_by_scene=True):
    """
    Log images to wandb organized by scene or as a flat gallery.
    
    Args:
        results: List of image info dicts
        wandb_project: W&B project name
        wandb_run_name: W&B run name (optional)
        group_by_scene: If True, group images by scene in wandb
    """
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=wandb_run_name or "inference-results",
        job_type="inference"
    )
    
    if group_by_scene:
        # Group by scene
        scenes = {}
        for result in results:
            scene = result["scene"]
            if scene not in scenes:
                scenes[scene] = []
            scenes[scene].append(result)
        
        # Log each scene separately
        for scene, scene_results in tqdm(scenes.items(), desc="Logging scenes"):
            images = []
            for result in scene_results:
                caption = f"{result['pair']} ({result['type']})"
                images.append(wandb.Image(result["path"], caption=caption))
            
            wandb.log({
                f"results/{scene}": images
            })
            print(f"✅ Logged {len(images)} images from {scene}")
    else:
        # Log all images in one gallery
        images = []
        for result in tqdm(results, desc="Preparing images"):
            caption = f"{result['scene']}/{result['pair']} ({result['type']})"
            images.append(wandb.Image(result["path"], caption=caption))
        
        wandb.log({
            "results/all": images
        })
        print(f"✅ Logged {len(images)} images")
    
    wandb.finish()


def log_images_as_table(results, wandb_project, wandb_run_name=None):
    """
    Log images to wandb as a table for easier browsing.
    
    Args:
        results: List of image info dicts
        wandb_project: W&B project name
        wandb_run_name: W&B run name (optional)
    """
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=wandb_run_name or "inference-results-table",
        job_type="inference"
    )
    
    # Create table
    table_data = []
    for result in tqdm(results, desc="Preparing table"):
        table_data.append([
            result["scene"],
            result["pair"],
            result["type"],
            wandb.Image(result["path"])
        ])
    
    # Create wandb table
    table = wandb.Table(
        columns=["Scene", "Pair", "Type", "Image"],
        data=table_data
    )
    
    wandb.log({
        "results/table": table
    })
    
    print(f"✅ Logged {len(table_data)} images as table")
    wandb.finish()


def log_images_by_scene_separate(results, wandb_project, wandb_run_name_prefix="scene"):
    """
    Log images grouped by scene, with each scene as a separate wandb run.
    Useful for comparing scenes side-by-side in wandb UI.
    
    Args:
        results: List of image info dicts
        wandb_project: W&B project name
        wandb_run_name_prefix: Prefix for run names
    """
    # Group by scene
    scenes = {}
    for result in results:
        scene = result["scene"]
        if scene not in scenes:
            scenes[scene] = []
        scenes[scene].append(result)
    
    # Log each scene as a separate run
    for idx, (scene, scene_results) in enumerate(tqdm(scenes.items(), desc="Logging scenes")):
        # For multiple runs, we need to finish previous run and start new one
        if idx > 0:
            wandb.finish()
        
        wandb.init(
            project=wandb_project,
            name=f"{wandb_run_name_prefix}-{scene}",
            job_type="inference"
        )
        
        # Group by type within scene
        comparison_images = []
        inpainted_images = []
        
        for result in scene_results:
            caption = f"{result['pair']}"
            img = wandb.Image(result["path"], caption=caption)
            
            if result["type"] == "comparison":
                comparison_images.append(img)
            else:
                inpainted_images.append(img)
        
        log_dict = {}
        if comparison_images:
            log_dict[f"results/comparison"] = comparison_images
        if inpainted_images:
            log_dict[f"results/inpainted"] = inpainted_images
        
        wandb.log(log_dict)
        print(f"✅ Logged {len(scene_results)} images from {scene}")
    
    # Finish the last run
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Log inference results to wandb")
    
    # Required arguments
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing inference results (with scene/pair structure)")
    
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, required=True,
                        help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (optional)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/team name (optional)")
    
    # Filtering arguments
    parser.add_argument("--image_type", type=str, default="both",
                        choices=["both", "comparison", "inpainted"],
                        help="Type of images to log")
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                        help="Specific scenes to log (default: all)")
    
    # Logging mode
    parser.add_argument("--log_as_table", action="store_true",
                        help="Log images as a wandb table (better for browsing)")
    parser.add_argument("--log_by_scene_separate", action="store_true",
                        help="Log each scene as a separate wandb run")
    parser.add_argument("--group_by_scene", action="store_true", default=True,
                        help="Group images by scene (default: True)")
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory does not exist: {args.results_dir}")
        return
    
    # Find all result images
    print(f"🔍 Scanning results directory: {args.results_dir}")
    results = find_result_images(
        args.results_dir,
        image_type=args.image_type,
        scenes=args.scenes
    )
    
    if not results:
        print("❌ No result images found!")
        return
    
    print(f"✅ Found {len(results)} images to log")
    
    # Set wandb entity if provided
    if args.wandb_entity:
        os.environ["WANDB_ENTITY"] = args.wandb_entity
    
    # Log based on mode
    if args.log_by_scene_separate:
        print("📊 Logging each scene as a separate wandb run...")
        log_images_by_scene_separate(
            results,
            args.wandb_project,
            wandb_run_name_prefix=args.wandb_run_name or "scene"
        )
    elif args.log_as_table:
        print("📊 Logging images as wandb table...")
        log_images_as_table(
            results,
            args.wandb_project,
            args.wandb_run_name
        )
    else:
        print("📊 Logging images as gallery...")
        log_images_as_gallery(
            results,
            args.wandb_project,
            args.wandb_run_name,
            group_by_scene=args.group_by_scene
        )
    
    print("✅ Done!")


if __name__ == "__main__":
    main()

