#!/bin/bash
#SBATCH --job-name=tsm_resume
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bfid-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_resume_%j.out
#SBATCH --error=logs/train_resume_%j.err

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate mri_mrf_pytorch_env

cd $SLURM_SUBMIT_DIR
python scripts/train.py \
    --data_path data/tsm_50000_80hz.h5 \
    --run_dir runs/tsm_80hz \
    --epochs 100 \
    --batch_size 24 \
    --width 48 \
    --modes 16 \
    --in_channels 4 \
    --lambda_pde 0.0 \
    --lambda_acoustic 0.0 \
    --lambda_eps 1.0 \
    --lambda_ring 0.5 \
    --lambda_expand 0.2 \
    --resume
