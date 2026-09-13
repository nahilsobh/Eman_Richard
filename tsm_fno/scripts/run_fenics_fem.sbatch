#!/bin/bash
#SBATCH --job-name=ogden_fem
#SBATCH --account=bfid-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --output=/u/sobh/Eman_Richard/tsm_fno/results/fenics_fem_%j.log
#SBATCH --error=/u/sobh/Eman_Richard/tsm_fno/results/fenics_fem_%j.err

set -euo pipefail

# ── Environment setup ────────────────────────────────────────────────
export LD_LIBRARY_PATH=/u/sobh/.conda/envs/fenicsx/lib:${LD_LIBRARY_PATH:-}
PY=/u/sobh/.conda/envs/fenicsx/bin/python

# MUMPS/OpenMP threading — take full advantage of the requested CPUs
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "=== SLURM job info ==="
echo "  job_id       = $SLURM_JOB_ID"
echo "  partition    = $SLURM_JOB_PARTITION"
echo "  node         = $SLURM_JOB_NODELIST"
echo "  cpus/task    = $SLURM_CPUS_PER_TASK"
echo "  OMP threads  = $OMP_NUM_THREADS"
echo "  memory       = $SLURM_MEM_PER_NODE MB"
echo "  start        = $(date)"
echo

cd /u/sobh/Eman_Richard

# ── Resolution schedule ──────────────────────────────────────────────
# Interactive queue has a 1-hour cap. We progressively coarsen from the
# ideal 1 mm target down to a resolution that fits the wall-clock. Each
# stage prints its results before moving on — if 1 mm times out mid-run
# the coarser results are already saved.

DX_LIST=${DX_LIST:-"5 3"}       # can be overridden via `sbatch --export=DX_LIST="1"`
LOAD_STEPS=${LOAD_STEPS:-5}

for DX in $DX_LIST; do
    echo "==================================================================="
    echo "  Starting FEM run at dx = $DX mm"
    echo "==================================================================="
    T0=$(date +%s)
    $PY -u tsm_fno/scripts/paper_ogden_fenics_fem.py \
         --dx-mm $DX --load-steps $LOAD_STEPS || {
        echo "FEM run FAILED at dx=$DX mm (see error log)"
        continue
    }
    T1=$(date +%s)
    DT=$(( T1 - T0 ))
    echo "  → dx=$DX mm completed in $DT seconds"
    echo
    # Also render the summary figure
    $PY -u tsm_fno/scripts/paper_ogden_fenics_plot.py --dx-mm $DX || true
done

echo "=== end $(date) ==="
