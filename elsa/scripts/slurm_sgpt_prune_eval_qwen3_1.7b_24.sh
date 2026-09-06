#!/bin/bash
#SBATCH --job-name=sgpt_qwen3_1.7b_24
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91,n61,n64
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# SparseGPT 2:4 semi-structured prune + full eval for Qwen3-1.7B → reasoning_qwen3_1.7b
# Companion to the unstructured s50 run (wandb r21mokbt)
# Usage: sbatch slurm_sgpt_prune_eval_qwen3_1.7b_24.sh [NSAMPLES=128]

NSAMPLES=${1:-128}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SAVE_PATH="/home1/doyoonkim/projects/elsa/models/qwen3_1.7b_sgpt_s24_n${NSAMPLES}"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/slurm"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "=== SparseGPT Qwen3-1.7B (2:4 semi-structured) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "NSAMPLES=$NSAMPLES"
echo "SAVE_PATH=$SAVE_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/RAC/open-r1-main

$PYTHON src/open_r1/prune_and_eval.py \
    --model_path "$MODEL" \
    --sparsity 0.5 \
    --prune_n 2 \
    --prune_m 4 \
    --nsamples "$NSAMPLES" \
    --seqlen 2048 \
    --save_path "$SAVE_PATH" \
    --wandb_project reasoning_qwen3_1.7b \
    --wandb_name "sgpt_s24_ot_fw" \
    --push_to_hub

echo "##### END #####"
