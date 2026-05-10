#!/bin/bash
#SBATCH --job-name=mre_gen
#SBATCH --partition=cpu
#SBATCH --account=bfid-delta-cpu
#SBATCH --array=0-99
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err

module purge
conda activate mre_pipeline

python scripts/generate_chunk.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --chunk_size 500 \
    --grid_size 64 \
    --output_dir data/chunks
