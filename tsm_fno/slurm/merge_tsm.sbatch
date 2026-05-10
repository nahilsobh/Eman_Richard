#!/bin/bash
#SBATCH --job-name=tsm_merge
#SBATCH --partition=cpu
#SBATCH --account=bfid-delta-cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate mri_mrf_pytorch_env

cd $SLURM_SUBMIT_DIR
python scripts/merge_chunks.py
