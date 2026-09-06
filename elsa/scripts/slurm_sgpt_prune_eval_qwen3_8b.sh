#!/bin/bash
#SBATCH --job-name=sgpt_qwen3_8b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/sgpt_qwen3_8b_%j.out
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64
exec 2>&1

# SparseGPT prune + full eval for Qwen3-8B
# Usage: sbatch slurm_sgpt_prune_eval_qwen3_8b.sh <SPARSITY> [NSAMPLES=128] [CALIB=default|selfgen]

SPARSITY=${1:?"Usage: sbatch slurm_sgpt_prune_eval_qwen3_8b.sh <SPARSITY> [NSAMPLES] [CALIB=default|selfgen]"}
NSAMPLES=${2:-128}
# CALIB=selfgen -> calibrate on THIS model's own CoT traces (v3) mixed 80/20
# with FineWeb-Edu, the same JSONL ALPS calibrates on, instead of
# prune_and_eval.py's default OpenThoughts3-80%/FineWeb-Edu-20% built from
# HuggingFace. Anything else (default) keeps the original behavior.
CALIB=${3:-default}
SELFGEN_DATA="/home1/doyoonkim/projects/elsa/data/selfgen_ot3_fineweb_qwen3_8b_8192_v3.jsonl"
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
MODEL=$(ls -d /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ 2>/dev/null | head -1)
MODEL="${MODEL%/}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL/config.json" ]; then
    echo "ERROR: Qwen3-8B not found in HF cache" >&2
    exit 1
fi
SAVE_PATH="/home1/doyoonkim/projects/elsa/models/qwen3_8b_sgpt_s${SPARSITY_PCT}pct_n${NSAMPLES}${CALIB_TAG}"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/lighteval"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TMPDIR=/tmp

echo "=== SparseGPT Qwen3-8B (s${SPARSITY_PCT}%, n=${NSAMPLES}) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL"
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
    --wandb_project reasoning_qwen3_8b \
    --wandb_name "qwen3_8b_sgpt_s${SPARSITY_PCT}${CALIB_TAG}" \
    --push_to_hub

echo "##### END #####"
