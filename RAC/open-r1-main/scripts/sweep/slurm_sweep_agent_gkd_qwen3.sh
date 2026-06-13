#!/bin/bash
#SBATCH --job-name=gkd_sweep
#SBATCH --partition=A6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j.out
#SBATCH -t 3-00:00:00

# Usage: sbatch scripts/slurm_sweep_agent_gkd_qwen3.sh <SWEEP_ID>

SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not provided"
    echo "Usage: sbatch scripts/slurm_sweep_agent_gkd_qwen3.sh <SWEEP_ID>"
    exit 1
fi

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/RAC/open-r1-main

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting wandb agent for sweep: $SWEEP_ID"
/home1/doyoonkim/miniconda3/envs/rac/bin/wandb agent \
    dyk6208-gwangju-institute-of-science-and-technology/rac_qwen3_0.6_pruning/${SWEEP_ID}

echo "##### END #####"
