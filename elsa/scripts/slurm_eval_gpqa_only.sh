#!/bin/bash
#SBATCH --job-name=eval_gpqa_only
#SBATCH --partition=RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_gpqa_only_%j.out
exec 2>&1

# Backfill gpqa-only for a checkpoint whose gpqa eval failed elsewhere
# (e.g. log_cluster job 41546's vLLM crash on gpqa specifically, everything
# else already succeeded). Downloads the model straight from the HF repo
# it was already pushed to -- no local checkpoint needed.
#
# Usage: sbatch slurm_eval_gpqa_only.sh <HF_REPO_ID> <RUN_NAME> <WANDB_PROJECT> <SPARSITY>

HF_REPO=${1:?"Usage: sbatch slurm_eval_gpqa_only.sh <HF_REPO_ID> <RUN_NAME> <WANDB_PROJECT> <SPARSITY>"}
RUN_NAME=${2:?"missing RUN_NAME"}
WANDB_PROJECT=${3:?"missing WANDB_PROJECT"}
SPARSITY=${4:?"missing SPARSITY"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

LOCAL_JOB_BASE="/local-data/user-data/${USER}/eval_gpqa_only_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/eval_out" "$LOCAL_JOB_BASE/wandb"

DEBUG_COPY_DIR="/home1/doyoonkim/projects/elsa/logs/eval_gpqa_only_${SLURM_JOB_ID}_debug"
mkdir -p "$DEBUG_COPY_DIR"
copy_log_on_exit() {
    cp "$LOCAL_JOB_BASE/eval_out/eval_summary.json" "$DEBUG_COPY_DIR/" 2>/dev/null || true
}
trap copy_log_on_exit EXIT

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export WANDB_START_METHOD=fork
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export TMPDIR=/tmp

echo "=== gpqa-only backfill eval ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "HF_REPO=$HF_REPO"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname), cannot download $HF_REPO. Exiting." >&2
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$HF_REPO" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --method tr_gmp \
    --sparsity "$SPARSITY" \
    --skip_ppl \
    --skip_zeroshot \
    --benchmarks gpqa \
    --gpu_util 0.85 \
    --profile quick \
    --out_base "$LOCAL_JOB_BASE/eval_out"

EXIT_CODE=$?
echo "=== eval_full.py exit code: $EXIT_CODE ==="
echo "##### END #####"
