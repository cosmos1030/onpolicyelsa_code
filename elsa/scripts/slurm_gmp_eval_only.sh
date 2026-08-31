#!/bin/bash
#SBATCH --job-name=gmp_eval
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/local-data/user-data/%u/gmp_eval_%j/slurm_%j.out
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
exec 2>&1

MODEL_PATH=${1:?"Usage: sbatch slurm_gmp_eval_only.sh <model_path> <train_wbid> <sparsity> [profile]"}
TRAIN_WBID=${2:?"Need train_wbid (training run ID in reasoning_qwen3_4b)"}
SPARSITY=${3:?"Need sparsity"}
PROFILE=${4:-quick}  # quick=8192 budget (matches the rest of the dashboard) -- eval_full.py itself defaults to 'official'=32768, which does NOT match the 8192-budget baselines everywhere else on the dashboard unless explicitly overridden here.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
LOCAL_JOB_BASE="/local-data/user-data/${USER}/gmp_eval_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/eval_out"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL_PATH  TRAIN_WBID=$TRAIN_WBID  SPARSITY=$SPARSITY"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: model not found at $MODEL_PATH"
    exit 1
fi

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project reasoning_qwen3_4b \
    --method gmp \
    --sparsity "$SPARSITY" \
    --gpu_util 0.85 \
    --tp_size 1 \
    --profile "$PROFILE" \
    --out_base "$LOCAL_JOB_BASE/eval_out"

echo "##### END #####"
