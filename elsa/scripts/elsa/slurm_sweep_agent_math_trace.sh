#!/bin/bash
#SBATCH --job-name=elsa_sweep_trace
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j.out
#SBATCH -t 3-00:00:00

SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not provided"
    exit 1
fi

mkdir -p /home1/doyoonkim/projects/elsa/output_qwen

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Node: $(hostname)"
echo "Starting wandb sweep agent: ${SWEEP_ID}"
/home1/doyoonkim/miniconda3/envs/rac/bin/wandb agent \
    dyk6208-gwangju-institute-of-science-and-technology/elsa_qwen3_0.6/${SWEEP_ID}

echo "##### END #####"
