#!/bin/bash
#SBATCH --job-name=safe_prune_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/local-data/user-data/%u/safe_prune_1.7b_%j/slurm_%j.out
#SBATCH --exclude=n3,n42,n51,n52,n54,n55,n58,n60,n76,n77,n80
exec 2>&1

# SAFE pruning for Qwen3-1.7B — S50/60/70
# Usage: sbatch slurm_safe_prune_1.7b.sh <SPARSITY>

SPARSITY=${1:?"Usage: sbatch slurm_safe_prune_1.7b.sh <SPARSITY>"}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SAVE_DIR="/home1/doyoonkim/projects/elsa/models"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/safe_prune_1.7b_${SLURM_JOB_ID}"
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

echo "=== SAFE Pruning Qwen3-1.7B s${SPARSITY_PCT}% ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

TRAIN_LOG="$LOCAL_JOB_BASE/train_stdout.txt"

# ── SAFE Pruning ──────────────────────────────────────────────────────────────
$PYTHON main.py \
    --model="$MODEL" \
    --dataset=math_cot \
    --data_path="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_20k.jsonl" \
    --nsamples=128 \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_safe=true \
    --safe_lr=2e-4 \
    --safe_lmda=1e-3 \
    --safe_rho=0.05 \
    --safe_epochs=30 \
    --safe_warmup_epochs=2 \
    --safe_interval=32 \
    --safe_batch_size=4 \
    --safe_accumulation_steps=1 \
    --admm_beta1=0.9 \
    --admm_beta2=0.999 \
    --save_model=true \
    --admm_save_path="$SAVE_DIR" \
    --eval_math500=false \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42 \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit code: $TRAIN_EXIT ==="

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: SAFE pruning failed with exit $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

# ── Find saved model ──────────────────────────────────────────────────────────
SAVED_MODEL=$(grep -oP "(?<=Saved SAFE pruned model to )\S+" "$TRAIN_LOG" | tail -1)
if [ -z "$SAVED_MODEL" ] || [ ! -f "$SAVED_MODEL/config.json" ]; then
    SAVED_MODEL=$(ls -td ${SAVE_DIR}/Qwen3-1.7B_safe_s${SPARSITY_PCT}pct* 2>/dev/null | head -1)
    echo "WARNING: log parse failed, using most recent: $SAVED_MODEL"
fi
echo "=== Saved model: $SAVED_MODEL ==="

if [ -z "$SAVED_MODEL" ]; then
    echo "ERROR: Cannot find saved model path"
    exit 1
fi

# ── Full eval ─────────────────────────────────────────────────────────────────
echo "=== Starting eval_full.py ==="
$PYTHON scripts/eval_full.py \
    --model_path "$SAVED_MODEL" \
    --wandb_project reasoning_qwen3_1.7b \
    --run_name "safe_s${SPARSITY_PCT}pct" \
    --method safe \
    --sparsity ${SPARSITY} \
    --gpu_util 0.9 \
    --tp_size 1 \
    --out_base "$LOCAL_JOB_BASE/eval_out"

echo "##### END #####"
