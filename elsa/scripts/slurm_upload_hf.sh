#!/bin/bash
#SBATCH --job-name=hf_upload
#SBATCH --partition=A6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --qos=hpgpu
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/hf_upload_%j.out
#SBATCH --exclude=n3,n42,n60,n76,n77,n80
exec 2>&1

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

echo "=== HF Upload Job ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "Testing internet..."
curl -s --max-time 10 https://huggingface.co > /dev/null && echo "Internet OK" || echo "Internet FAIL"

cd /home1/doyoonkim/projects/elsa
$PYTHON scripts/upload_missing_hf_models.py

echo "##### END #####"
