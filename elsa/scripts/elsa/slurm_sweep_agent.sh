#!/bin/bash
#SBATCH --job-name=elsa_sweep
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=output/%j.out
#SBATCH -t 3-00:00:00

# Usage: sbatch scripts/slurm_sweep_agent.sh <SWEEP_ID>
# Submit N times to run N configs in parallel:
#   for i in {1..3}; do sbatch scripts/slurm_sweep_agent.sh <SWEEP_ID>; done

SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not provided"
    echo "Usage: sbatch slurm_sweep_agent.sh <SWEEP_ID>"
    exit 1
fi

mkdir -p output

cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting wandb agent for sweep: $SWEEP_ID"
/home1/doyoonkim/miniconda3/envs/elsa/bin/wandb agent \
    dyk6208-gwangju-institute-of-science-and-technology/elsa_qwen3_0.6/${SWEEP_ID}

echo "##### END #####"
