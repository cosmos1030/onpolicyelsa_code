#!/bin/bash
#SBATCH --job-name=eval_math500_7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --exclude=n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out

exec 2>&1

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb

export WANDB_DIR=/local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || grep WANDB_API_KEY ~/.bashrc | cut -d= -f2)
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

MODEL_PATH="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"

cd /home1/doyoonkim/projects/elsa

/home1/doyoonkim/miniconda3/envs/rac/bin/python main.py \
    --model=$MODEL_PATH \
    --dataset=mixed_cot \
    --data_path=/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
    --sparsity_ratio=0.0 \
    --do_gmp=false \
    --do_grpo_opkd=false \
    --eval_math500=true \
    --math500_model_path=$MODEL_PATH \
    --math500_max_new_tokens=8192 \
    --math500_max_samples=500 \
    --eval_zero_shot=false \
    --save_model=false \
    --wandb=true \
    --wandb_project=gmp_qwen3_7b \
    --push_to_hub=false \
    --seed=42
