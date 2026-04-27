#!/bin/bash
#SBATCH --job-name=elsa_deepseek_ntp
#SBATCH --partition=H200-ZT-PCIe
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_deepseek/%j.out
#SBATCH -t 3-00:00:00

SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not provided"
    exit 1
fi

mkdir -p /home1/doyoonkim/projects/elsa/output_deepseek

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "Starting wandb sweep agent: ${SWEEP_ID}"
/home1/doyoonkim/miniconda3/envs/rac/bin/wandb agent \
    dyk6208-gwangju-institute-of-science-and-technology/elsa_deepseek_1.5b/${SWEEP_ID}

echo "##### END #####"
