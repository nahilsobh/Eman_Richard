#!/bin/bash
#SBATCH --account=bfid-delta-cpu
#SBATCH --partition=cpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --job-name=vec_yindim_rigidball
#SBATCH --output=/u/sobh/Eman_Richard/tsm_fno/results/vec_yindim_rigidball_%j.log
#SBATCH --error=/u/sobh/Eman_Richard/tsm_fno/results/vec_yindim_rigidball_%j.err

# Vector Navier + tensor µ_ij + rigid balloon + multi-face broadband + curl→MIP
# at Yin's 18 x 15 x 15 cm container dimensions (NZ,NY,NX) = (60,50,50)
# at dx = 3 mm — matches Yin's acquisition voxel exactly.

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
