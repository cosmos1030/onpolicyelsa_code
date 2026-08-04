#!/bin/bash
#SBATCH --job-name=gmp_fsdp_smoke_4b
#SBATCH --partition=A6000
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/gmp_fsdp_smoke_4b_%j.out
#SBATCH --exclude=n3,n42,n60,n76,n77,n80
exec 2>&1

# Multi-GPU GMP NTP+KD smoke test (FSDP 2 GPU, 64 steps) + full eval (5 samples/bench)
# Usage: sbatch slurm_gmp_ntp_kd_fsdp_smoke_4b.sh [SPARSITY=0.5]

SPARSITY=${1:-0.5}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"
SCRIPT_DIR="/home1/doyoonkim/projects/elsa"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/eval_out"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_CACHE="/home1/doyoonkim/.cache/huggingface/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE

echo "=== GMP NTP+KD FSDP 2-GPU smoke test: Qwen3-4B s${SPARSITY_PCT}% 64 steps ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$SCRIPT_DIR"

TRAIN_LOG="$LOCAL_JOB_BASE/train_stdout.txt"

# ── Training (torchrun 2 GPU, FSDP) ─────────────────────────────────────────
$TORCHRUN --nproc_per_node=2 --master_port=29501 main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --gmp_steps=5 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=1 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=5 \
    --gmp_fisher_beta=0.999 \
    --gmp_max_seq_len=1024 \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=0.5 \
    --gmp_kd_lambda=0.5 \
    --gmp_prompt_path="$DATA_PATH" \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b \
    --seed=42 \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit code: $TRAIN_EXIT ==="

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit $TRAIN_EXIT" >&2
    exit $TRAIN_EXIT
fi

# Extract saved model path from training log
SAVED_MODEL=$(grep "Saved pruned model to" "$TRAIN_LOG" | tail -1 | awk '{print $NF}')

if [ -z "$SAVED_MODEL" ] || [ ! -f "$SAVED_MODEL/config.json" ]; then
    # Fallback: find most recent dir in models/
    SAVED_MODEL=$(ls -td /home1/doyoonkim/projects/elsa/models/gmp_* 2>/dev/null | head -1)
    echo "WARNING: log parse failed, using most recent: $SAVED_MODEL"
fi

echo "=== Saved model: $SAVED_MODEL ==="

if [ -z "$SAVED_MODEL" ]; then
    echo "ERROR: Cannot find saved model path" >&2
    exit 1
fi

# ── Full eval: PPL + zeroshot + lighteval bench (6 tasks) (5 samples each) ────────────
# Unset torchrun/torchelastic env vars that interfere with vLLM's dist init
unset TORCHELASTIC_WORKER_PORT TORCHELASTIC_ERROR_FILE TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS
unset MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE
echo "=== Starting eval_full.py (tp_size=2, max_samples=5) ==="

$PYTHON scripts/eval_full.py \
    --model_path "$SAVED_MODEL" \
    --wandb_project reasoning_qwen3_4b \
    --run_name "gmp_fsdp_smoke_s${SPARSITY_PCT}pct" \
    --method gmp \
    --sparsity ${SPARSITY} \
    --gpu_util 0.85 \
    --tp_size 2 \
    --max_samples 5 \
    --out_base "$LOCAL_JOB_BASE/eval_out"

echo "##### END #####"
