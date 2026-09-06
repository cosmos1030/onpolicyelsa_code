#!/bin/bash
#SBATCH --job-name=sgpt_qwen3_1.7b
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91,n61,n64
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# SparseGPT unstructured prune + full eval for Qwen3-1.7B → reasoning_qwen3_1.7b
# Usage: sbatch slurm_sgpt_prune_eval_qwen3_1.7b.sh <SPARSITY> [NSAMPLES=128] [CALIB=default|selfgen]

SPARSITY=${1:?"Usage: sbatch slurm_sgpt_prune_eval_qwen3_1.7b.sh <SPARSITY> [NSAMPLES] [CALIB=default|selfgen]"}
NSAMPLES=${2:-128}
# CALIB=selfgen -> calibrate on THIS model's own CoT traces (v3) mixed 80/20
# with FineWeb-Edu, the same JSONL ALPS calibrates on, instead of
# prune_and_eval.py's default OpenThoughts3-80%/FineWeb-Edu-20% built from
# HuggingFace. Anything else (default) keeps the original behavior.
CALIB=${3:-default}
SELFGEN_DATA="/home1/doyoonkim/projects/elsa/data/selfgen_ot3_fineweb_qwen3_8192_v3.jsonl"
if [ "$CALIB" = "selfgen" ]; then
    if [ ! -f "$SELFGEN_DATA" ]; then
        echo "ERROR: self-gen calibration file not found: $SELFGEN_DATA"; exit 1
    fi
    CALIB_ARG="--calib_data_path $SELFGEN_DATA"
    CALIB_TAG="_selfgenv3"
else
    CALIB_ARG=""
    CALIB_TAG=""
fi
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SAVE_PATH="/home1/doyoonkim/projects/elsa/models/qwen3_1.7b_sgpt_s${SPARSITY_PCT}pct_n${NSAMPLES}${CALIB_TAG}"

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

echo "=== SparseGPT Qwen3-1.7B (s${SPARSITY_PCT}%) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "SPARSITY=$SPARSITY  NSAMPLES=$NSAMPLES"
echo "SAVE_PATH=$SAVE_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/RAC/open-r1-main

$PYTHON src/open_r1/prune_and_eval.py \
    --model_path "$MODEL" \
    --sparsity "$SPARSITY" \
    --nsamples "$NSAMPLES" \
    $CALIB_ARG \
    --seqlen 2048 \
    --save_path "$SAVE_PATH" \
    --wandb_project reasoning_qwen3_1.7b \
    --wandb_name "sgpt_s${SPARSITY_PCT}${CALIB_TAG}" \
    --push_to_hub

echo "##### END #####"
