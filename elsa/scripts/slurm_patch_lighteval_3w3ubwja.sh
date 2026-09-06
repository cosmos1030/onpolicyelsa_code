#!/bin/bash
#SBATCH --job-name=patch_3w3ubwja
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
#SBATCH -t 0-04:00:00
#SBATCH --exclude=n3,n60,n80

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "${LOCAL_JOB_BASE}/slurm"

echo "Node: $(hostname)"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(cat ~/.wandb_api_key 2>/dev/null || cat ~/.netrc 2>/dev/null | grep -A1 'api.wandb.ai' | tail -1 | awk '{print $2}' || echo "")
export HF_HOME="/local-data/user-data/${USER}/hf_cache"
export VLLM_USE_V1=0

MODEL_PATH="/home1/doyoonkim/projects/elsa/models/gmp_s60pct_lr0.0001_20260705_113727"

cd /home1/doyoonkim/projects/elsa
exec 2>&1

/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/patch_lighteval_wandb.py \
    --model_path "${MODEL_PATH}" \
    --wandb_run_id "3w3ubwja" \
    --out_dir "${LOCAL_JOB_BASE}/lighteval_patch" \
    --gpu_util 0.85 \
    --project reasoning_pruning_v2

echo "=== done ==="
