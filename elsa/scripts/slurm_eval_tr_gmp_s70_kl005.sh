#!/bin/bash
#SBATCH --job-name=eval_tr_s70
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

MODEL_PATH="/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_20260707_011522"

cd /home1/doyoonkim/projects/elsa

exec 2>&1

/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/eval_full.py \
    --model_path "${MODEL_PATH}" \
    --wandb_project reasoning_pruning_v2 \
    --run_name "tr_gmp_ntp_kd_s70_kl0005_eval" \
    --method gmp_tr \
    --sparsity 0.70 \
    --gpu_util 0.85 \
    --out_base "${LOCAL_JOB_BASE}/eval_out"

echo "=== eval done ==="
