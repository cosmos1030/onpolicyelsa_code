#!/bin/bash
#SBATCH --job-name=flash_vs_sdpa
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/elsa/debug_flash_vs_sdpa_%j.out
#SBATCH -t 0-00:15:00

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)"
/home1/doyoonkim/miniconda3/envs/rac/bin/python -u debug_flash_vs_sdpa.py
echo "##### END #####"
