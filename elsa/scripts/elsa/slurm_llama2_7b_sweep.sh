#!/bin/bash
#SBATCH --job-name=elsa_llama2_7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --output=output/%j.out
#SBATCH -t 3-00:00:00

# Usage: sbatch scripts/slurm_llama2_7b_sweep.sh <SWEEP_ID>
# Submit N times to run N configs in parallel:
#   for i in {1..3}; do sbatch scripts/slurm_llama2_7b_sweep.sh <SWEEP_ID>; done

SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not provided"
    echo "Usage: sbatch scripts/slurm_llama2_7b_sweep.sh <SWEEP_ID>"
    exit 1
fi

mkdir -p output

cd /home1/doyoonkim/projects/elsa

echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "CUDA_HOME=$CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

echo "Starting wandb agent for sweep: $SWEEP_ID"
/home1/doyoonkim/miniconda3/envs/elsa/bin/wandb agent \
    dyk6208-gwangju-institute-of-science-and-technology/elsa_llama2_7b/${SWEEP_ID}

date
echo "##### END #####"
