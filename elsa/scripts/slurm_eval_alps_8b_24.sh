#!/bin/bash
#SBATCH --job-name=eval_alps_8b_24
#SBATCH --partition=RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_alps_8b_24_%j.out
exec 2>&1

# Re-eval-only for the already-pruned Qwen3-8B ALPS 2:4 checkpoint (job 719641
# pruned it fine, but its reasoning-bench eval failed: --tp_size 2 was passed
# with only --gres=gpu:1 allocated, so Ray's placement group request for 2
# GPUs timed out and all 5 lighteval subprocesses died with exit code 1).
# Checkpoint is untouched at qwen3_8b_alps_s24, so we just re-run eval_full.py
# on a single GPU with tp_size=1 instead of re-pruning from scratch.

MODEL_PATH="/home1/doyoonkim/projects/elsa/models/qwen3_8b_alps_s24"

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: model not found at $MODEL_PATH" >&2
    exit 1
fi

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

LOCAL_JOB_BASE="/local-data/user-data/${USER}/eval_alps_8b_24_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/eval_out" "$LOCAL_JOB_BASE/wandb"

DEBUG_COPY_DIR="/home1/doyoonkim/projects/elsa/logs/eval_alps_8b_24_${SLURM_JOB_ID}_debug"
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
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export TMPDIR=/tmp

echo "=== Full eval (retry): ALPS Qwen3-8B 2:4 semi-structured ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL_PATH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project reasoning_qwen3_8b \
    --run_name "alps_8b_s24_reeval" \
    --method alps \
    --sparsity 0.5 \
    --gpu_util 0.85 \
    --tp_size 1 \
    --profile quick \
    --out_base "$LOCAL_JOB_BASE/eval_out"

EXIT_CODE=$?
echo "=== eval_full.py exit code: $EXIT_CODE ==="
echo "##### END #####"
