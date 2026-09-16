#!/bin/bash
#SBATCH --account=bfid-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --job-name=vec_N60_rigidball
#SBATCH --output=/u/sobh/Eman_Richard/tsm_fno/results/vec_N60_rigidball_%j.log
#SBATCH --error=/u/sobh/Eman_Richard/tsm_fno/results/vec_N60_rigidball_%j.err

# Vector Navier + tensor µ_ij + rigid balloon + multi-face broadband + curl→MIP
# at N=60 (dx=3 mm, matching Yin's acquisition voxel).

set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-64}

echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "Threads: OMP=$OMP_NUM_THREADS  MKL=$MKL_NUM_THREADS"
echo "Started: $(date)"

cd /u/sobh/Eman_Richard/tsm_fno

/u/sobh/.conda/envs/mri_mrf_pytorch_env/bin/python \
    scripts/paper_wave_sim_sobh_vector_mip_multiface.py

echo "Finished: $(date)"
