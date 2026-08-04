#!/bin/bash
#SBATCH --job-name=mmlu_full_dense
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/mmlu_full_dense_%j.out
#SBATCH -t 4:00:00
#SBATCH --exclude=n3,n51,n54,n60,n76,n80

exec 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export WANDB_START_METHOD=thread
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "Node: $(hostname), Job: $SLURM_JOB_ID"
date

mkdir -p /home1/doyoonkim/projects/elsa/eval_outputs/dense_1.5b_mmlu_full
mkdir -p /home1/doyoonkim/projects/elsa/logs

/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/test_mmlu_full_dense.py

echo "##### END #####"
date
