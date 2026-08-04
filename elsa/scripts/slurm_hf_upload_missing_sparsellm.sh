#!/bin/bash
#SBATCH --job-name=hf_upload_sparsellm
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/hf_upload_sparsellm_%j.out
exec 2>&1

# Retroactive HF Hub upload for SparseLLM 4B s70 (model saved, HF link never
# captured). CPU-only (plain file upload, no GPU).

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/hf_upload_missing_sparsellm.py

echo "##### END #####"
