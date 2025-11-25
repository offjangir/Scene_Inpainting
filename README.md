# 3D Scene Inpainting with ControlNet

A PyTorch implementation for training and inference of ControlNet-based inpainting models on 3D scene data. This project supports both Stable Diffusion 1.5 (SD 1.5) and Stable Diffusion XL (SDXL) architectures.

## Overview

This project implements ControlNet-based inpainting models that can fill in missing regions (holes) in 3D scene images. The models are trained to condition on binary masks and masked images to generate realistic completions.

### Key Features

- **Multiple Model Support**: SD 1.5 and SDXL variants
- **Flexible Training**: Train from scratch or fine-tune pretrained ControlNet models
- **Batch Inference**: Process entire directories of images
- **WandB Integration**: Automatic logging of training metrics and visualizations
- **Checkpoint Management**: Automatic checkpoint saving and resuming

## Project Structure

```
3d_inpainting/
├── configs/                    # Training configuration files
│   ├── train_sd15_inpaint.yaml  # SD 1.5 training config
│   ├── train_sdxl_inpaint.yaml  # SDXL training config
│   └── ...
├── utils/                      # Shared utilities
│   ├── env_setup.py            # Environment and cache setup
│   ├── dataset.py              # Shared dataset class
│   └── __init__.py
├── train_sd15_inpaint.py       # SD 1.5 training script
├── train_sxl_inpaint_v2.py     # SDXL training script (v2)
├── inference.py                 # SDXL inference script
├── inference_sd15_inpaint.py   # SD 1.5 inference script
├── run_inference.sh            # Batch inference script
├── sbatch.bash                  # SLURM job submission script
└── README.md                    # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.13+ with CUDA support
- CUDA-capable GPU (recommended: A6000 or better)

### Setup

1. Clone the repository:
```bash
cd /data/user_data/yjangir/3d_inpainting
```

2. Install dependencies:
```bash
pip install torch torchvision diffusers transformers accelerate
pip install wandb pillow numpy tqdm pyyaml matplotlib
pip install xformers  # Optional, for memory-efficient attention
```

3. Set up environment variables (update paths as needed):
```bash
export HF_HOME=/scratch/$USER/hf_cache
export HUGGINGFACE_HUB_CACHE=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export TORCH_HOME=/scratch/$USER/torch_cache
```

## Dataset Format

The training scripts expect data organized in the following structure:

```
dataset/
├── scene1/
│   ├── pair1/
│   │   ├── image_target.png    # Ground truth complete image
│   │   ├── image_cond.png      # Masked/incomplete image
│   │   └── mask_cond.png       # Binary mask (1=hole, 0=valid)
│   ├── pair2/
│   │   └── ...
│   └── ...
├── scene2/
│   └── ...
└── ...
```

**Mask Convention**:
- `invert_mask=False` (default): 1 = hole to inpaint, 0 = valid region
- `invert_mask=True`: 0 = hole to inpaint, 1 = valid region

## Training

### SD 1.5 Training

Train or fine-tune a ControlNet model on SD 1.5:

```bash
python train_sd15_inpaint.py configs/train_sd15_inpaint.yaml
```

**Key Features**:
- Uses pretrained inpainting ControlNet (`lllyasviel/control_v11p_sd15_inpaint`)
- Supports checkpoint resuming (set `resume_from_checkpoint: auto` in config)
- Automatic validation and visualization

### SDXL Training

Train a ControlNet model on SDXL:

```bash
python train_sxl_inpaint_v2.py configs/train_config.yaml
```

**Key Features**:
- Trains ControlNet from scratch (mask-only conditioning)
- Supports separate train/test datasets
- Random validation sampling for diversity

### Configuration

Edit the YAML config files in `configs/` to customize:

- **Training parameters**: epochs, batch size, learning rate
- **Data paths**: training and test dataset directories
- **Model settings**: base model, scheduler, checkpoint paths
- **Logging**: WandB project name, validation frequency

Example config structure:
```yaml
train:
  epochs: 1000
  batch_size: 8
  lr: 0.00001
  output_dir: ./output/checkpoints

data:
  root: ./dataset_bridge
  test_root: ./test_data
  size: 512
  invert_mask: false

model:
  pretrained_model: runwayml/stable-diffusion-v1-5
  pretrained_controlnet: lllyasviel/control_v11p_sd15_inpaint
```

### SLURM Submission

For cluster training, use the provided SLURM script:

```bash
sbatch sbatch.bash
```

Edit `sbatch.bash` to customize GPU requirements, time limits, and environment setup.

## Inference

### Single Image Inference

**SDXL**:
```bash
python inference.py \
    --controlnet_path output/controlnet_final \
    --mask_path path/to/mask.png \
    --cond_path path/to/conditioning.png \
    --output_path output/result.png \
    --num_steps 50 \
    --guidance_scale 7.5 \
    --seed 42
```

**SD 1.5**:
```bash
python inference_sd15_inpaint.py \
    --controlnet_path output/controlnet_final \
    --mask_path path/to/mask.png \
    --cond_path path/to/conditioning.png \
    --output_path output/result.png
```

### Batch Inference

Process entire directories:

```bash
python inference.py \
    --controlnet_path output/controlnet_final \
    --input_dir dataset/test \
    --output_dir output/results \
    --batch \
    --size 512 \
    --num_steps 50 \
    --save_comparison
```

Or use the provided script:
```bash
bash run_inference.sh
```

### Inference Parameters

- `--controlnet_path`: Path to trained ControlNet (`.pt` file or diffusers directory)
- `--base_model`: Base model identifier (default: `stabilityai/stable-diffusion-xl-base-1.0`)
- `--num_steps`: Number of diffusion steps (default: 50)
- `--guidance_scale`: Classifier-free guidance scale (default: 7.5)
- `--seed`: Random seed for reproducibility
- `--size`: Processing image size (default: 512)
- `--invert_mask`: Invert mask convention if needed
- `--fp16`: Use float16 precision (faster, less accurate)

## Model Architectures

### SD 1.5 Approach
- **Base Model**: Stable Diffusion 1.5 (frozen)
- **ControlNet**: Pretrained inpainting ControlNet (fine-tuned)
- **Input**: Masked RGB image (3 channels)
- **Conditioning**: RGB image with holes marked white

### SDXL Approach (V2)
- **Base Model**: Stable Diffusion XL (frozen)
- **ControlNet**: Trained from scratch
- **Input**: Mask only (1 channel, padded to 4)
- **Conditioning**: Mask for ControlNet, masked image latents for UNet

## Checkpoints

Checkpoints are saved in two formats:

1. **PyTorch format**: `controlnet_epoch{N}.pt` - Model state dict
2. **Diffusers format**: `controlnet_epoch{N}/` - Full diffusers-compatible directory

The training script automatically saves checkpoints every N epochs (configurable) and supports resuming from the last checkpoint.

## Monitoring

### WandB Integration

Training metrics and visualizations are automatically logged to WandB:

- Training loss, learning rate, gradient norms
- Validation samples and losses
- Test set evaluations

Configure WandB in the config file:
```yaml
logging:
  wandb_project: my-project
  wandb_run_name: my-run
  wandb_resume: allow
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce batch size in config
   - Use `--fp16` flag for inference
   - Enable CPU offloading (if supported)

2. **NaN Losses**:
   - Check data for corrupted images
   - Reduce learning rate
   - Enable gradient clipping (already enabled)

3. **Slow Training**:
   - Increase `num_workers` in DataLoader
   - Use xformers for memory-efficient attention
   - Enable mixed precision training

4. **Checkpoint Loading Errors**:
   - Ensure checkpoint format matches model architecture
   - Check file paths in config
   - Verify model state dict keys match

## File Descriptions

- `train_sd15_inpaint.py`: SD 1.5 training with pretrained ControlNet fine-tuning
- `train_sxl_inpaint_v2.py`: SDXL training from scratch (mask-only)
- `inference.py`: SDXL inference script with batch support
- `inference_sd15_inpaint.py`: SD 1.5 inference script
- `quick_inference_test.py`: Quick test script for single samples
- `log_results_to_wandb.py`: Utility to log inference results to WandB
- `sbatch.bash`: SLURM job submission script
- `run_inference.sh`: Batch inference automation script

## Citation

If you use this code, please cite:

```bibtex
@software{3d_inpainting,
  title = {3D Scene Inpainting with ControlNet},
  author = {Your Name},
  year = {2024}
}
```

## License

[Specify your license here]

## Contact

For questions or issues, please open an issue or contact [your email].

