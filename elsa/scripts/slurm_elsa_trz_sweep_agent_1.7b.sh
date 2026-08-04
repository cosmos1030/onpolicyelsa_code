#!/bin/bash
#SBATCH --job-name=elsa_trz_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/local-data/user-data/%u/elsa_trz_1.7b_%j/slurm_%j.out
exec 2>&1

# ELSA TR-z projection sweep agent for Qwen3-1.7B
# Usage: for i in {1..9}; do sbatch slurm_elsa_trz_sweep_agent_1.7b.sh <SWEEP_ID>; done

SWEEP_ARG=${1:?"Usage: sbatch slurm_elsa_trz_sweep_agent_1.7b.sh <SWEEP_ID>"}

if [[ "$SWEEP_ARG" != *"/"* ]]; then
    SWEEP_ARG="dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_1.7b/${SWEEP_ARG}"
fi

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
LOCAL_JOB_BASE="/local-data/user-data/${USER}/elsa_trz_1.7b_${SLURM_JOB_ID}"
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

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  SWEEP=$SWEEP_ARG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Internet check — wandb agent needs internet to fetch sweep config
if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname) — cannot run wandb sweep agent. Exiting."
    exit 1
fi
echo "Internet OK on $(hostname)"

cd /home1/doyoonkim/projects/elsa

TRAIN_LOG="$LOCAL_JOB_BASE/train_stdout.txt"

# ── Training: run exactly 1 sweep config ─────────────────────────────────────
/home1/doyoonkim/miniconda3/envs/rac/bin/wandb agent --count 1 "$SWEEP_ARG" \
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
    SAVED_MODEL=$(ls -td /home1/doyoonkim/projects/elsa/models/Qwen3-1.7B_pruned* 2>/dev/null | head -1)
    echo "WARNING: log parse failed, using most recent: $SAVED_MODEL"
fi
echo "=== Saved model: $SAVED_MODEL ==="

if [ -z "$SAVED_MODEL" ]; then
    echo "ERROR: Cannot find saved model path"
    exit 1
fi

# ── Extract training wandb run ID ─────────────────────────────────────────────
TRAIN_RUN_ID=$(grep -oP "(?<=runs/)[a-z0-9]{8}(?=[^/]|$)" "$TRAIN_LOG" 2>/dev/null | tail -1)
echo "=== Training wandb run ID: ${TRAIN_RUN_ID:-NOT FOUND} ==="

# ── Full eval: PPL + zeroshot + reasoning (math500) ──────────────────────────
echo "=== Starting eval_full.py ==="

EVAL_ARGS=(
    --model_path "$SAVED_MODEL"
    --wandb_project reasoning_qwen3_1.7b
    --method elsa-trz
    --gpu_util 0.9
    --tp_size 1
    --out_base "$LOCAL_JOB_BASE/eval_out"
)
if [ -n "$TRAIN_RUN_ID" ]; then
    EVAL_ARGS+=(--wandb_run_id "$TRAIN_RUN_ID")
fi

$PYTHON scripts/eval_full.py "${EVAL_ARGS[@]}"

echo "##### END #####"
