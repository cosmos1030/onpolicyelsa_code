#!/bin/bash
#SBATCH --job-name=alps_sft_4b
#SBATCH --partition=RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31
#SBATCH --output=/local-data/user-data/%u/alps_sft_4b_%j/slurm_%j.out
exec 2>&1

# ALPS (one-shot pruned) -> fixed-mask NTP sparse SFT for Qwen3-4B, OT80/FW20.
# Loads the already-pruned ALPS checkpoint, freezes its zero pattern
# (gmp_fixed_mask=true skips Fisher-based mask updates), and continues
# training with plain NTP loss only (gmp_kd_lambda=0) — same budget as the
# ELSA plain rerun: steps=4096, global batch=16 (batch_size=1 x grad_accum=4 x
# world_size=4), lr=1e-4 cosine with 256-step warmup, seqlen=2048.
#
# 4x RTX6000ADA FSDP (gmp_use_fsdp=true): a first single-GPU attempt
# (691120/691121) genuinely CUDA-OOM'd on step 1 — full fp32 Adam states for
# a 4B model don't fit in one 48GB card without FSDP sharding, same lesson as
# ELSA plain 4B needing multi-GPU (see feedback_gpu_sizing memory).
#
# Usage: sbatch slurm_alps_sparse_sft_qwen3_4b.sh <SPARSITY> [SPARSITY_TYPE]
#   sparsity=0.7 -> qwen3_4b_alps_s70pct
#   sparsity=0.5 sparsity_type=2:4 -> qwen3_4b_alps_s24
# e.g.: sbatch slurm_alps_sparse_sft_qwen3_4b.sh 0.7
#       sbatch slurm_alps_sparse_sft_qwen3_4b.sh 0.5 2:4

SPARSITY=${1:?"Usage: sbatch slurm_alps_sparse_sft_qwen3_4b.sh <SPARSITY> [SPARSITY_TYPE]"}
SPARSITY_TYPE=${2:-unstructured}

if [ "$SPARSITY_TYPE" = "2:4" ]; then
    ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_s24"
else
    SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
    ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_s${SPARSITY_PCT}pct"
fi
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun

LOCAL_JOB_BASE="/local-data/user-data/${USER}/alps_sft_4b_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

DEBUG_LOG_COPY="/home1/doyoonkim/projects/elsa/logs/alps_sft_4b_${SLURM_JOB_ID}_last.out"
copy_log_on_exit() { cp "$LOCAL_JOB_BASE/slurm_${SLURM_JOB_ID}.out" "$DEBUG_LOG_COPY" 2>/dev/null || true; }
trap copy_log_on_exit EXIT

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
export TMPDIR=/tmp
export NCCL_DEBUG=WARN
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== ALPS -> Sparse SFT (NTP, fixed mask) Qwen3-4B s${SPARSITY_PCT:-24} (${SPARSITY_TYPE}), 4xRTX6000ADA FSDP ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$ALPS_MODEL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=4 --master_port=${MASTER_PORT} main.py \
    --model="$ALPS_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=${SPARSITY_TYPE} \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --gmp_use_fsdp=true \
    --steps=4096 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --lr=1e-4 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --seqlen=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=1.0 \
    --gmp_kd_lambda=0.0 \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b \
    --seed=42

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
