#!/bin/bash
# Inference script for warped test dataset
# Run with: bash run_inference.sh

# Activate the right conda environment (update this if needed)
# conda activate datagen  # or whatever environment has diffusers installed

cd /data/user_data/yjangir/3d_inpainting

# Run inference using the latest SDXL checkpoint
python scripts/inference/inference.py \
  --controlnet_path /data/group_data/katefgroup/datasets/yjangir/3d_inpaint/checkpoint_more_data/controlnet_epoch42 \
  --base_model stabilityai/stable-diffusion-xl-base-1.0 \
  --input_dir /data/user_data/yjangir/3d_inpainting/dataset_gene_warped_test \
  --output_dir /data/user_data/yjangir/3d_inpainting/inference_results_sdxl_epoch42 \
  --batch \
  --size 512 \
  --num_steps 50 \
  --guidance_scale 7.5 \
  --seed 42 \
  --save_comparison

echo "✅ Inference complete! Results saved to inference_results_sdxl_epoch42"

