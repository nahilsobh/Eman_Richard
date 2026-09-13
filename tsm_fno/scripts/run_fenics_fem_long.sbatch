#!/bin/bash
#SBATCH --job-name=ogden_fem_long
#SBATCH --account=bfid-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --output=/u/sobh/Eman_Richard/tsm_fno/results/fenics_fem_long_%j.log
#SBATCH --error=/u/sobh/Eman_Richard/tsm_fno/results/fenics_fem_long_%j.err

set -euo pipefail

export LD_LIBRARY_PATH=/u/sobh/.conda/envs/fenicsx/lib:${LD_LIBRARY_PATH:-}
PY=/u/sobh/.conda/envs/fenicsx/bin/python

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "=== SLURM job info ==="
echo "  job_id      = $SLURM_JOB_ID"
echo "  partition   = $SLURM_JOB_PARTITION"
echo "  node        = $SLURM_JOB_NODELIST"
echo "  cpus/task   = $SLURM_CPUS_PER_TASK"
echo "  OMP threads = $OMP_NUM_THREADS"
echo "  memory      = $SLURM_MEM_PER_NODE MB"
echo "  start       = $(date)"
echo

cd /u/sobh/Eman_Richard

DX_LIST=${DX_LIST:-"3 2 1"}
LOAD_STEPS=${LOAD_STEPS:-10}

# Solver auto-selection: LU for < ~500k DOFs, CG+GAMG above
declare -A SOLVER_MAP
SOLVER_MAP[3]="lu"
SOLVER_MAP[2]="lu"
SOLVER_MAP[1]="cg-gamg"

for DX in $DX_LIST; do
    SOLVER=${SOLVER_MAP[$DX]:-cg-gamg}
    echo "==================================================================="
    echo "  dx = $DX mm, solver = $SOLVER"
    echo "==================================================================="
    T0=$(date +%s)
    $PY -u tsm_fno/scripts/paper_ogden_fenics_fem_quarter.py \
         --dx-mm $DX --load-steps $LOAD_STEPS --solver $SOLVER \
        || echo "FEM run FAILED at dx=$DX mm (see error log)"
    T1=$(date +%s)
    DT=$(( T1 - T0 ))
    echo "  → dx=$DX mm completed in $DT seconds ($(( DT / 60 )) min)"
    echo
done

echo "=== end $(date) ==="
