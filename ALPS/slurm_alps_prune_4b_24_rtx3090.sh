#!/bin/bash
#SBATCH --job-name=alps_4b_24_rtx3090
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/local-data/user-data/%u/alps_4b_24_rtx3090_%j/slurm_%j.out
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
exec 2>&1

# ALPS 2:4 semi-structured pruning + full eval for Qwen3-4B
# Usage: sbatch slurm_alps_prune_4b_24.sh

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
SAVE_BASE="/home1/doyoonkim/projects/elsa/models"
SAVED_MODEL="${SAVE_BASE}/qwen3_4b_alps_s24"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/alps_4b_24_rtx3090_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/eval_out"

# Node access is revoked once the job ends, so /local-data logs/results become
# permanently unreachable if something (network outage, wandb failure) goes
# wrong — copy the log and eval_summary.json to NFS once on exit (small, one-
# time write, not continuous I/O) so post-mortem/recovery is possible.
DEBUG_COPY_DIR="/home1/doyoonkim/projects/elsa/logs/alps_4b_24_rtx3090_${SLURM_JOB_ID}_debug"
mkdir -p "$DEBUG_COPY_DIR"
copy_log_on_exit() {
    cp "$LOCAL_JOB_BASE/slurm_${SLURM_JOB_ID}.out" "$DEBUG_COPY_DIR/" 2>/dev/null || true
    cp "$LOCAL_JOB_BASE/eval_out/eval_summary.json" "$DEBUG_COPY_DIR/" 2>/dev/null || true
}
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
unset HF_HUB_OFFLINE
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

echo "=== ALPS Pruning Qwen3-4B 2:4 semi-structured ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/ALPS

$PYTHON qwen3_alps.py \
    "$MODEL" \
    0.5 \
    --data_path "$DATA" \
    --nsamples 128 \
    --nm_n 2 \
    --nm_m 4 \
    --rho 300.0 \
    --seed 42 \
    --save "$SAVED_MODEL" \
    --eval_full \
    --wandb_project reasoning_qwen3_4b \
    --run_name "alps_s24" \
    --gpu_util 0.9 \
    --tp_size 1 \
    --out_base "$LOCAL_JOB_BASE/eval_out" \
    --profile quick \
    --push_to_hub

EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE ==="
echo "##### END #####"
