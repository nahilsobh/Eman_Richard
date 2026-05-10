#!/bin/bash
#SBATCH --job-name=mre_train
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --account=bfid-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate mri_mrf_pytorch_env

cd $SLURM_SUBMIT_DIR
python src/train_fno.py \
    --data_path data/mre_v3_50000.h5 \
    --run_dir runs/phase0_v3
