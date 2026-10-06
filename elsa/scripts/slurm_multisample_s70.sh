#!/bin/bash
#SBATCH --job-name=multisample_s70
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/multisample_%j.out
exec 2>&1

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

echo "=== multisample within-model correct-vs-wrong length check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON /home1/doyoonkim/projects/elsa/scripts/multisample_run.py \
    "cosmos1030/gmp-kd3e-1-s70pct-lr5e-5_20260811_115604" \
    "${1:-10}" \
    "${2:-4}" \
    "${3:-/home1/doyoonkim/projects/elsa/models/multisample_s70_debug.json}"

echo "=== DONE ==="
