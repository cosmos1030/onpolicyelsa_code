#!/bin/bash
#SBATCH --job-name=prune_24
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --exclude=n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out

exec 2>&1

SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not provided"
    echo "Usage: sbatch scripts/slurm_sweep_agent_deepseek_24_sparsegpt.sh <SWEEP_ID>"
    exit 1
fi

_SAVED_CUDA=${CUDA_VISIBLE_DEVICES:-}
ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
[ -n "$_SAVED_CUDA" ] && export CUDA_VISIBLE_DEVICES="$_SAVED_CUDA"

if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/slurm"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/RAC/open-r1-main

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export WANDB_DISABLE_SERVICE=1
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export WANDB_DIR="$LOCAL_JOB_BASE/wandb"

echo "Node: $(hostname), SWEEP: $SWEEP_ID"

/home1/doyoonkim/miniconda3/envs/rac/bin/wandb agent \
    dyk6208-gwangju-institute-of-science-and-technology/gmp_24_semi_structured/${SWEEP_ID}

echo "##### END #####"
