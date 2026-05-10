#!/bin/bash
#SBATCH --job-name=tsm_gen
#SBATCH --partition=cpu
#SBATCH --account=bfid-delta-cpu
#SBATCH --array=0-99
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err

source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate mri_mrf_pytorch_env

cd $SLURM_SUBMIT_DIR
python scripts/generate_chunk.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --chunk_size 500 \
    --grid_size 80 \
    --output_dir data/chunks_80hz
