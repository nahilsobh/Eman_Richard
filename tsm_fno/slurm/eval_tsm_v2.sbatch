#!/bin/bash
#SBATCH --job-name=tsm_eval_v2
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bfid-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_v2_%j.out
#SBATCH --error=logs/eval_v2_%j.err

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate mri_mrf_pytorch_env

cd $SLURM_SUBMIT_DIR
python scripts/evaluate.py \
    --checkpoint runs/tsm_v2/best.pt \
    --data_path data/tsm_50000.h5 \
    --save_dir results/tsm_v2
