#!/bin/bash
#SBATCH --job-name=gmp_ntp_kd_8k_8b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/local-data/user-data/%u/gmp_ntp_kd_8k_8b_%j/slurm_%j.out
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
exec 2>&1

# GMP NTP+KD for Qwen3-8B, 8192 steps, FSDP 2-GPU
# Usage: sbatch slurm_gmp_ntp_kd_8192_qwen3_8b.sh <SPARSITY>

SPARSITY=${1:?"Usage: sbatch slurm_gmp_ntp_kd_8192_qwen3_8b.sh <SPARSITY>"}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL=$(ls -d /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ 2>/dev/null | head -1)
MODEL="${MODEL%/}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL/config.json" ]; then
    echo "ERROR: Qwen3-8B not found in HF cache" >&2
    exit 1
fi
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/gmp_ntp_kd_8k_8b_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/eval_out"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE

echo "=== GMP NTP+KD Qwen3-8B s${SPARSITY_PCT}% 8192 steps ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

TRAIN_LOG="$LOCAL_JOB_BASE/train_stdout.txt"

# ── Training (torchrun 2 GPU, FSDP) ─────────────────────────────────────────
$TORCHRUN --nproc_per_node=4 --master_port=29501 main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --gmp_steps=8192 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=32 \
    --gmp_fisher_beta=0.999 \
    --gmp_max_seq_len=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=0.5 \
    --gmp_kd_lambda=0.5 \
    --gmp_prompt_path="$DATA_PATH" \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_zero_shot=true \
    --eval_full_bench=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_8b \
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
    SAVED_MODEL=$(ls -td /home1/doyoonkim/projects/elsa/models/gmp_* 2>/dev/null | head -1)
    echo "WARNING: log parse failed, using most recent: $SAVED_MODEL"
fi

echo "=== Saved model: $SAVED_MODEL ==="

if [ -z "$SAVED_MODEL" ]; then
    echo "ERROR: Cannot find saved model path" >&2
    exit 1
fi

# ── Full eval: PPL + zeroshot + lighteval bench (6 tasks) ─────────────────────────────
unset TORCHELASTIC_WORKER_PORT TORCHELASTIC_ERROR_FILE TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS
unset MASTER_ADDR MASTER_PORT RANK LOCAL_RANK WORLD_SIZE
echo "=== Starting eval_full.py (tp_size=2) ==="

$PYTHON scripts/eval_full.py \
    --model_path "$SAVED_MODEL" \
    --wandb_project reasoning_qwen3_8b \
    --run_name "gmp_ntp_kd_s${SPARSITY_PCT}" \
    --method gmp \
    --sparsity ${SPARSITY} \
    --gpu_util 0.85 \
    --tp_size 4 \
    --out_base "$LOCAL_JOB_BASE/eval_out"

echo "##### END #####"
