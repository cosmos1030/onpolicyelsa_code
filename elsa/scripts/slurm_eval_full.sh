#!/bin/bash
#SBATCH --job-name=eval_full
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n76,n77,n80,n87,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_full_%j.out
exec 2>&1

# Full eval (PPL + zero-shot + bench (6 tasks)) for an existing checkpoint
# Usage: sbatch slurm_eval_full.sh <MODEL_PATH> <RUN_NAME> <METHOD> <SPARSITY> <WANDB_PROJECT>
# e.g.:  sbatch slurm_eval_full.sh /path/to/model sgpt_s60 sparsegpt 0.6 reasoning_qwen3_4b

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_full.sh <MODEL_PATH> <RUN_NAME> <METHOD> <SPARSITY> <WANDB_PROJECT>"}
RUN_NAME=${2:-"eval"}
METHOD=${3:-"sparsegpt"}
SPARSITY=${4:-0.0}
WANDB_PROJECT=${5:-"reasoning_pruning_v2"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="/home1/doyoonkim/projects/elsa/logs/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY=$(grep "^export WANDB_API_KEY=" ~/.bashrc 2>/dev/null | cut -d= -f2-)
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_CACHE="/home1/doyoonkim/.cache/huggingface/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== eval_full: $RUN_NAME ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL_PATH=$MODEL_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --method "$METHOD" \
    --sparsity "$SPARSITY" \
    --gpu_util 0.85 \
    --out_base "$LOCAL_JOB_BASE/eval_${RUN_NAME}"

echo "##### END #####"
