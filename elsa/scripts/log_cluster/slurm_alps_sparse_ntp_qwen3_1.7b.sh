#!/bin/bash
#SBATCH --job-name=alps_sparse_ntp_1.7b
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/alps_sparse_ntp_1.7b_%j.out
exec 2>&1

# ALPS (one-shot pruned, s50pct) -> fixed-mask NTP-only sparse SFT, Qwen3-1.7B,
# OT80/FW20 (cosmos1030/ot3-fineweb-200k-qwen3). Loads the already-pruned ALPS
# checkpoint, freezes its zero pattern (gmp_fixed_mask=true skips Fisher-based
# mask updates -- pure sparse fine-tuning, no further pruning) and trains with
# NTP loss only (gmp_ntp_lambda=1.0 default, gmp_kd_lambda=0.0 default, no
# on-policy KD) -- same budget/loss-mix pattern as
# rerun_ot80fw20/slurm_alps_sparse_sft_qwen3_4b.sh, adapted to this cluster
# and to the 1.7B single-GPU sizing from slurm_alps_sft_ntpkd_opkd_qwen3_1.7b.sh.
#
# This cluster (log-node01-07 / log-master) has no /local-data scratch, so
# wandb writes go to node-local /tmp instead. HF_HOME points at the shared
# team cache (/home/shared/huggingface, see log_efficient_qwen_competition
# scripts) -- the Qwen3-1.7B base tokenizer/config may already be there, but
# the ALPS checkpoint and dataset aren't, so this stays in online mode (no
# *_OFFLINE=1) rather than assuming a fully warm cache.
#
# Usage: sbatch slurm_alps_sparse_ntp_qwen3_1.7b.sh [SPARSITY] [LR_SCHEDULER]
# e.g.: sbatch slurm_alps_sparse_ntp_qwen3_1.7b.sh
#       sbatch slurm_alps_sparse_ntp_qwen3_1.7b.sh 0.5 cosine

SPARSITY=${1:-0.5}
LR_SCHEDULER=${2:-cosine}

ALPS_MODEL="cosmos1030/alps-s50pct_20260802_055049"
REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
DATA_PATH="${REPO_ROOT}/data/ot3_fineweb_200k_qwen3_train.jsonl"

PYTHON=/home/doyoonkim/.conda/envs/rac/bin/python

LOCAL_JOB_BASE="/tmp/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/models"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_HOME=/home/shared/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_HOST_IP=127.0.0.1

echo "=== ALPS -> Sparse SFT NTP-only Qwen3-1.7B s${SPARSITY} lr_scheduler=${LR_SCHEDULER} (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$ALPS_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd "$REPO_ROOT"

$PYTHON main.py \
    --model="$ALPS_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=1e-4 \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --seqlen=2048 \
    --gmp_ntp_lambda=1.0 \
    --gmp_kd_lambda=0.0 \
    --gmp_save_path="${REPO_ROOT}/models" \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
