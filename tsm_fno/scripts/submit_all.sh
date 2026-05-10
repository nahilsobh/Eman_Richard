#!/usr/bin/env bash
set -e

mkdir -p logs data/chunks data runs results

GEN_ID=$(sbatch --parsable slurm/gen_tsm.sbatch)
echo "Submitted gen array      : $GEN_ID"

MERGE_ID=$(sbatch --parsable --dependency=afterok:$GEN_ID slurm/merge_tsm.sbatch)
echo "Submitted merge          : $MERGE_ID  (after $GEN_ID)"

TRAIN_ID=$(sbatch --parsable --dependency=afterok:$MERGE_ID slurm/train_tsm.sbatch)
echo "Submitted train          : $TRAIN_ID  (after $MERGE_ID)"

EVAL_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_ID slurm/eval_tsm.sbatch)
echo "Submitted eval           : $EVAL_ID  (after $TRAIN_ID)"

echo
echo "Dependency chain: $GEN_ID -> $MERGE_ID -> $TRAIN_ID -> $EVAL_ID"
echo "Monitor with: squeue --me"
