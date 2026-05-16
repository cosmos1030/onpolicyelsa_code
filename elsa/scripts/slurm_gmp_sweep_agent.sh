#!/bin/bash
#SBATCH --job-name=gmp_sweep
#SBATCH --partition=A6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
#SBATCH -t 3-00:00:00
#SBATCH --exclude=n3,n60,n76,n80

# Usage:
#   wandb sweep sweep_configs/gmp_ntp_deepseek_1.5b.yaml
#   for i in {1..4}; do sbatch scripts/slurm_gmp_sweep_agent.sh <SWEEP_ID>; done

# Accept either a full path (entity/project/sweep_id) or just a sweep ID.
# If just an ID, default to the v2 project.
SWEEP_ARG=$1
if [ -z "$SWEEP_ARG" ]; then
    echo "ERROR: SWEEP_ID not provided"
    exit 1
fi

# If no slashes, assume it's a bare ID and prepend entity/project
if [[ "$SWEEP_ARG" != *"/"* ]]; then
    SWEEP_ARG="dyk6208-gwangju-institute-of-science-and-technology/gmp_qwen3_1.5b_v2/${SWEEP_ARG}"
fi

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/wandb"

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_START_METHOD=thread

echo "Node: $(hostname), SWEEP: $SWEEP_ARG"

/home1/doyoonkim/miniconda3/envs/rac/bin/wandb agent "$SWEEP_ARG"

echo "##### END #####"
