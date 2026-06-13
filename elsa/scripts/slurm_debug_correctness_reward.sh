#!/bin/bash
#SBATCH --job-name=debug_crrwd
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --time=0-00:20:00
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out

exec 2>&1

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/slurm"

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_MODE=disabled

echo "Node: $(hostname)"
/home1/doyoonkim/miniconda3/envs/rac/bin/python -u debug_correctness_reward.py
echo "##### END #####"
