#!/bin/bash
#SBATCH --job-name=eval_zeroshot_only
#SBATCH --partition=RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_zeroshot_only_%j.out
exec 2>&1

# Backfill zero-shot only (9-task) for a checkpoint whose main run's
# zero-shot pass got cut off (e.g. by a post-hoc segfault at process exit
# after lighteval+PPL+HF-push already succeeded). skip_lighteval+skip_ppl
# so this is a quick single-pass HF-model eval, no vLLM subprocess needed.
#
# Usage: sbatch slurm_eval_zeroshot_only.sh <MODEL_PATH> <RUN_NAME> <WANDB_PROJECT> <SPARSITY>

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_zeroshot_only.sh <MODEL_PATH> <RUN_NAME> <WANDB_PROJECT> <SPARSITY>"}
RUN_NAME=${2:?"missing RUN_NAME"}
WANDB_PROJECT=${3:?"missing WANDB_PROJECT"}
SPARSITY=${4:?"missing SPARSITY"}

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: model not found at $MODEL_PATH" >&2
    exit 1
fi

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

LOCAL_JOB_BASE="/local-data/user-data/${USER}/eval_zeroshot_only_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/eval_out" "$LOCAL_JOB_BASE/wandb"

DEBUG_COPY_DIR="/home1/doyoonkim/projects/elsa/logs/eval_zeroshot_only_${SLURM_JOB_ID}_debug"
mkdir -p "$DEBUG_COPY_DIR"
copy_log_on_exit() {
    cp "$LOCAL_JOB_BASE/eval_out/eval_summary.json" "$DEBUG_COPY_DIR/" 2>/dev/null || true
}
trap copy_log_on_exit EXIT

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export WANDB_START_METHOD=fork
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp

echo "=== Zero-shot-only backfill eval ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL_PATH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --method tr_gmp \
    --sparsity "$SPARSITY" \
    --skip_lighteval \
    --skip_ppl \
    --out_base "$LOCAL_JOB_BASE/eval_out"

EXIT_CODE=$?
echo "=== eval_full.py exit code: $EXIT_CODE ==="
echo "##### END #####"
