#!/bin/bash
#SBATCH --job-name=fno1d_train
#SBATCH --partition=cpu-preempt
#SBATCH --account=bfid-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/logs/%j_train.out
#SBATCH --error=slurm/logs/%j_train.err

set -eo pipefail

cd /u/sobh/Eman_Richard/eigenstrain_1d

# Activate conda env (unset -u first — conda scripts use unbound vars)
set +u
source /sw/rh9.4/python/miniforge3/etc/profile.d/conda.sh
conda activate mri_mrf_pytorch_env
set -u

# Use all allocated CPUs for BLAS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start: $(date)"

python scripts/train_1d.py \
    --data_path   data/1d_pairs_20000.h5 \
    --run_dir     runs/fno_1d_full \
    --epochs      200 \
    --modes       32 \
    --width       64 \
    --n_layers    4 \
    --lr          1e-3 \
    --batch_size  64 \
    --lambda_true     1.0 \
    --lambda_analytic 0.1 \
    --lambda_pde      0.0 \
    --lambda_expand   0.2

echo "Done: $(date)"
