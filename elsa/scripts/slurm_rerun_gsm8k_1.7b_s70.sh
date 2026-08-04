#!/bin/bash
#SBATCH --job-name=rerun_gsm8k_1.7b_s70
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91,n61,n64
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/rerun_gsm8k_1.7b_s70_%j.out
exec 2>&1

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/rerun_gsm8k_missing.py \
    /home1/doyoonkim/projects/elsa/models/qwen3_1.7b_sgpt_s70pct_n128 \
    reasoning_qwen3_1.7b \
    y90smdu6 \
    /local-data/user-data/${USER}/job_${SLURM_JOB_ID}/gsm8k_rerun

echo "##### END #####"
