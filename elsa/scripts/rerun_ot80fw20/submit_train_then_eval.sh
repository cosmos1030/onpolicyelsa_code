#!/bin/bash
# Submit a training job, then chain a follow-up eval job to it via
# --dependency=afterok so eval starts automatically once training finishes
# successfully (and never runs at all if training fails/crashes).
#
# Usage: submit_train_then_eval.sh <TRAIN_SCRIPT> <EVAL_PARTITION> <EVAL_QOS> \
#          <WANDB_PROJECT> <METHOD> <SPARSITY> -- <TRAIN_SCRIPT_ARGS...>
#
# e.g.:
#   ./submit_train_then_eval.sh slurm_elsa_plain_qwen3_1.7b.sh RTX3090 normal \
#       reasoning_qwen3_1.7b elsa 0.6 -- 0.6 1e-3
#
#   ./submit_train_then_eval.sh slurm_elsa_plain_qwen3_8b.sh A100-80GB hpgpu \
#       reasoning_qwen3_8b elsa 0.7 -- 0.7 1e-4 5e-3

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_SCRIPT=${1:?"Usage: submit_train_then_eval.sh <TRAIN_SCRIPT> <EVAL_PARTITION> <EVAL_QOS> <WANDB_PROJECT> <METHOD> <SPARSITY> -- <TRAIN_SCRIPT_ARGS...>"}
EVAL_PARTITION=${2:?"missing EVAL_PARTITION"}
EVAL_QOS=${3:?"missing EVAL_QOS"}
WANDB_PROJECT=${4:?"missing WANDB_PROJECT"}
METHOD=${5:?"missing METHOD"}
SPARSITY=${6:?"missing SPARSITY"}
shift 6
if [ "$1" != "--" ]; then
    echo "ERROR: expected -- before TRAIN_SCRIPT_ARGS, got '$1'" >&2
    exit 1
fi
shift

TRAIN_OUT=$(sbatch "${SCRIPT_DIR}/${TRAIN_SCRIPT}" "$@")
echo "$TRAIN_OUT"
TRAIN_JOB_ID=$(echo "$TRAIN_OUT" | grep -oE '[0-9]+$')
if [ -z "$TRAIN_JOB_ID" ]; then
    echo "ERROR: could not parse train job ID from sbatch output" >&2
    exit 1
fi

EVAL_OUT=$(sbatch --dependency=afterok:${TRAIN_JOB_ID} \
    --partition="${EVAL_PARTITION}" --qos="${EVAL_QOS}" \
    "${SCRIPT_DIR}/slurm_eval_chained.sh" "${TRAIN_JOB_ID}" "${WANDB_PROJECT}" "${METHOD}" "${SPARSITY}")
echo "$EVAL_OUT"
echo "Train job ${TRAIN_JOB_ID} -> eval job chained (afterok), will start automatically once training succeeds."
