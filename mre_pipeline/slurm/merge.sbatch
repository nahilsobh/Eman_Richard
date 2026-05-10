#!/bin/bash
#SBATCH --job-name=mre_merge
#SBATCH --partition=cpu
#SBATCH --account=bfid-delta-cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --dependency=afterok:$GEN_JOB_ID
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err

module purge
conda activate mre_pipeline

python scripts/merge_chunks.py
