#!/bin/bash
#SBATCH --job-name=prune_deepseek_prompt
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs/%j.out
#SBATCH -t 3-00:00:00

# Usage:
#   wandb sweep sweep_configs/deepseek_r1_1.5b_prompt_sparsity.yaml
#   for i in {1..5}; do sbatch scripts/slurm_sweep_agent_deepseek_prompt_sparsity.sh <SWEEP_ID>; done

SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not provided"
    echo "Usage: sbatch scripts/slurm_sweep_agent_deepseek_prompt_sparsity.sh <SWEEP_ID>"
    exit 1
fi

mkdir -p /home1/doyoonkim/projects/RAC/open-r1-main/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/RAC/open-r1-main

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting wandb agent for sweep: $SWEEP_ID"
/home1/doyoonkim/miniconda3/envs/rac/bin/wandb agent \
    dyk6208-gwangju-institute-of-science-and-technology/gmp_qwen3_1.5b/${SWEEP_ID}

echo "##### END #####"
