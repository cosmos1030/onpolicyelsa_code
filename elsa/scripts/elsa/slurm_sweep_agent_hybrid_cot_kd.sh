#!/bin/bash
#SBATCH --job-name=elsa_hybrid_kd
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j.out
#SBATCH -t 3-00:00:00
#SBATCH --exclude=n3

# Usage:
#   1. Create sweep (one time):
#      wandb sweep config/sweep_qwen_hybrid_cot_kd.yaml
#      → prints: "wandb: Created sweep with ID: <SWEEP_ID>"
#
#   2. Submit 12 agents (4 lr × 3 lmda = 12 configs):
#      for i in {1..12}; do sbatch scripts/slurm_sweep_agent_hybrid_cot_kd.sh <SWEEP_ID>; done

SWEEP_ID=$1
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: SWEEP_ID not provided"
    echo "Usage: sbatch scripts/slurm_sweep_agent_hybrid_cot_kd.sh <SWEEP_ID>"
    exit 1
fi

mkdir -p /home1/doyoonkim/projects/elsa/output_qwen

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

echo "Node: $(hostname)"
echo "Starting wandb sweep agent: ${SWEEP_ID}"
/home1/doyoonkim/miniconda3/envs/rac/bin/wandb agent \
    dyk6208-gwangju-institute-of-science-and-technology/elsa_qwen3_0.6b/${SWEEP_ID}

echo "##### END #####"