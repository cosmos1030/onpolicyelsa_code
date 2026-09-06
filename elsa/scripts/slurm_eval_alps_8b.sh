#!/bin/bash
#SBATCH --job-name=eval_alps_8b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_alps_8b_%j.out
exec 2>&1

# Full eval (PPL + zero-shot + lighteval bench (6 tasks)) for saved ALPS Qwen3-8B checkpoint
# Usage: sbatch slurm_eval_alps_8b.sh <SPARSITY_PCT>
#   e.g. sbatch slurm_eval_alps_8b.sh 50
#        sbatch slurm_eval_alps_8b.sh 60

SPARSITY_PCT=${1:?"Usage: sbatch slurm_eval_alps_8b.sh <50|60|70>"}
MODEL_PATH="/home1/doyoonkim/projects/elsa/models/qwen3_8b_alps_s${SPARSITY_PCT}pct"

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: model not found at $MODEL_PATH" >&2
    exit 1
fi

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

LOCAL_JOB_BASE="/local-data/user-data/${USER}/eval_alps_8b_s${SPARSITY_PCT}_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/eval_out" "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export WANDB_START_METHOD=fork
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export TMPDIR=/tmp

echo "=== Full eval: ALPS Qwen3-8B s${SPARSITY_PCT}% ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL_PATH"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project reasoning_qwen3_4b \
    --run_name "alps_8b_s${SPARSITY_PCT}pct" \
    --method alps \
    --sparsity "0.${SPARSITY_PCT}" \
    --gpu_util 0.85 \
    --tp_size 2 \
    --out_base "$LOCAL_JOB_BASE/eval_out"

EXIT_CODE=$?
echo "=== eval_full.py exit code: $EXIT_CODE ==="
echo "##### END #####"
