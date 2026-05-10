#!/usr/bin/env bash
set -e

mkdir -p logs data/chunks data runs

GEN_JOB_ID=$(sbatch --parsable slurm/gen_array.sbatch)
echo "Submitted generation job: $GEN_JOB_ID"

MERGE_JOB_ID=$(GEN_JOB_ID=$GEN_JOB_ID sbatch --parsable \
    --dependency=afterok:$GEN_JOB_ID slurm/merge.sbatch)
echo "Submitted merge job: $MERGE_JOB_ID (depends on $GEN_JOB_ID)"

echo ""
echo "Dependency chain:"
echo "  gen_array [$GEN_JOB_ID] --> merge [$MERGE_JOB_ID]"
echo ""
echo "Estimated completion: ~1.5 hours from now"
echo "Monitor with: squeue --me"
