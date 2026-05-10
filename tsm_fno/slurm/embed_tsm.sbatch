#!/bin/bash
#SBATCH --job-name=tsm_embed
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bfid-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate mri_mrf_pytorch_env

cd $SLURM_SUBMIT_DIR
python scripts/extract_embeddings.py \
    --checkpoint runs/tsm_80hz/best.pt \
    --data_path data/tsm_50000_80hz.h5 \
    --save_dir results/tsm_80hz/embeddings
