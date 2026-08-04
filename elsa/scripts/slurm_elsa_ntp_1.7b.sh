#!/bin/bash
#SBATCH --job-name=elsa_ntp_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3-00:00:00
#SBATCH --output=/local-data/user-data/%u/elsa_ntp_1.7b_%j.out
#SBATCH --exclude=n3,n42,n51,n52,n54,n55,n58,n60,n76,n77,n80
exec 2>&1

# ELSA (ADMM NTP-only) baseline for Qwen3-1.7B with OT+FW data
# Usage: sbatch slurm_elsa_ntp_1.7b.sh [SPARSITY=0.5]
#   s50: lr=1e-1  s60: lr=1e-2  s70: lr=5e-3

SPARSITY=${1:-0.5}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

# lr per sparsity
if   [ "${SPARSITY_PCT}" = "50" ]; then ADMM_LR=1e-1
elif [ "${SPARSITY_PCT}" = "60" ]; then ADMM_LR=1e-2
elif [ "${SPARSITY_PCT}" = "70" ]; then ADMM_LR=5e-3
else
    echo "ERROR: Unsupported sparsity ${SPARSITY}. Use 0.5, 0.6, or 0.7."
    exit 1
fi

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"
SAVE_DIR="/home1/doyoonkim/projects/elsa/models"

ADMM_STEPS=4096
ADMM_LMDA=5e-5
ADMM_LMDA_SCHEDULE=cosine
ADMM_INTERVAL=32
BS=1
ACCUM=8

LOCAL_JOB_BASE="/local-data/user-data/${USER}/elsa_1.7b_job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE

echo "=== ELSA NTP Qwen3-1.7B s${SPARSITY_PCT}% lr=${ADMM_LR} (${ADMM_STEPS} steps) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

TRAIN_LOG="$LOCAL_JOB_BASE/train_stdout.txt"

# ── Training ──────────────────────────────────────────────────────────────────
$PYTHON main.py \
    --model="${MODEL}" \
    --dataset=mixed_cot \
    --data_path="${DATA_PATH}" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --admm_steps=${ADMM_STEPS} \
    --admm_batch_size=${BS} \
    --admm_gradient_accumulation_steps=${ACCUM} \
    --admm_lr=${ADMM_LR} \
    --admm_lmda=${ADMM_LMDA} \
    --admm_final_lmda=${ADMM_LMDA} \
    --admm_lmda_schedule_mode=${ADMM_LMDA_SCHEDULE} \
    --admm_beta2=0.999 \
    --admm_interval=${ADMM_INTERVAL} \
    --admm_base_optimizer=adamw \
    --admm_precision=bf16 \
    --save_model=True \
    --admm_save_path="${SAVE_DIR}" \
    --eval_zero_shot=False \
    --wandb=True \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42 \
    2>&1 | tee "$TRAIN_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}
echo "=== Training exit code: $TRAIN_EXIT ==="

if [ $TRAIN_EXIT -ne 0 ]; then
    echo "ERROR: Training failed with exit $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

# ── Find saved model ──────────────────────────────────────────────────────────
SAVED_MODEL=$(grep -oP "(?<=Saved pruned model to )\S+" "$TRAIN_LOG" | tail -1)
if [ -z "$SAVED_MODEL" ] || [ ! -f "$SAVED_MODEL/config.json" ]; then
    SAVED_MODEL=$(ls -td ${SAVE_DIR}/Qwen3-1.7B_pruned${SPARSITY}* 2>/dev/null | head -1)
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
    --run_name "elsa_ntp_s${SPARSITY_PCT}pct_lr${ADMM_LR}_lmda${ADMM_LMDA}" \
    --method elsa \
    --sparsity ${SPARSITY} \
    --gpu_util 0.9

echo "##### END #####"
