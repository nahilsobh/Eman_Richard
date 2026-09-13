#!/bin/bash
#SBATCH --job-name=ogden_mn
#SBATCH --account=bfid-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=/u/sobh/Eman_Richard/tsm_fno/results/fenics_mnewton_%j.log
#SBATCH --error=/u/sobh/Eman_Richard/tsm_fno/results/fenics_mnewton_%j.err

set -euo pipefail
export LD_LIBRARY_PATH=/u/sobh/.conda/envs/fenicsx/lib:${LD_LIBRARY_PATH:-}
PY=/u/sobh/.conda/envs/fenicsx/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "=== SLURM job info ==="
echo "  job_id      = $SLURM_JOB_ID"
echo "  partition   = $SLURM_JOB_PARTITION"
echo "  node        = $SLURM_JOB_NODELIST"
echo "  cpus/task   = $SLURM_CPUS_PER_TASK"
echo "  OMP threads = $OMP_NUM_THREADS"
echo "  start       = $(date)"
echo

cd /u/sobh/Eman_Richard

DX=${DX:-3}
MAX_IT=${MAX_IT:-100}
RTOL=${RTOL:-1e-6}

echo "==================================================================="
echo "  Modified Newton FEM at dx = $DX mm"
echo "==================================================================="
T0=$(date +%s)
$PY -u tsm_fno/scripts/paper_ogden_fenics_mnewton.py \
     --dx-mm $DX --max-it $MAX_IT --rtol $RTOL \
    || echo "FEM run FAILED at dx=$DX mm (see error log)"
T1=$(date +%s)
DT=$(( T1 - T0 ))
echo "  → completed in $DT seconds ($(( DT / 60 )) min)"

echo "=== end $(date) ==="
