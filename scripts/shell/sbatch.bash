#!/bin/bash
#SBATCH --job-name=3d_inpaint_train
#SBATCH --cpus-per-task=16                  # Adjust as needed, 16 is reasonable for one GPU
#SBATCH --mem=64G                           # Fit to your training needs
#SBATCH --gres=gpu:A6000:1                  # 1 GPU
#SBATCH --time=24:00:00                     # Training time, adjust as needed
#SBATCH --partition=general
#SBATCH --output=/data/user_data/yjangir/3d_inpainting/slurm/output/%x_%j.log
#SBATCH --error=/data/user_data/yjangir/3d_inpainting/slurm/output/%x_%j.log
#SBATCH --mail-type=END
#SBATCH --mail-user=yjangir@andrew.cmu.edu

# ----------- Environment setup -----------
# export TORCH_SHOW_CPP_STACKTRACES=1
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdxl-inpaint

cd /data/user_data/yjangir/3d_inpainting/

export HF_HOME=/scratch/$USER/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export DIFFUSERS_CACHE=$HF_HOME/diffusers
export HF_HUB_CACHE=$HF_HOME/hub
export XDG_CACHE_HOME=$HF_HOME


# ----------- Log directory -----------
LOGDIR="/data/user_data/yjangir/3d_inpainting/slurm"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/train_${SLURM_JOB_ID}.log"
touch "$LOGFILE"

echo "Logging output to: $LOGFILE"
echo "Job started on $(date)" >> "$LOGFILE"

# ----------- Launch training -----------
python scripts/train/train_sd15_inpaint.py >"$LOGFILE" 2>&1
