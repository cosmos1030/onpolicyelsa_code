#!/bin/bash
#SBATCH --job-name=corr_grpo
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

SWEEP_ARG=$1
if [ -z "$SWEEP_ARG" ]; then
    echo "ERROR: SWEEP_ID not provided"
    exit 1
fi
if [[ "$SWEEP_ARG" != *"/"* ]]; then
    SWEEP_ARG="dyk6208-gwangju-institute-of-science-and-technology/gmp_qwen3_1.5b_v2/${SWEEP_ARG}"
fi

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/slurm"

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | tail -1 | awk -F= '{print $2}')
export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_START_METHOD=thread

echo "Node: $(hostname), SWEEP: $SWEEP_ARG"

/home1/doyoonkim/miniconda3/envs/rac/bin/wandb agent "$SWEEP_ARG"

echo "##### END #####"
