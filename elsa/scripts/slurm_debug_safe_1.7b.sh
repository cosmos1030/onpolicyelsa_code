#!/bin/bash
#SBATCH --job-name=safe_debug
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:30:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/debug_safe_%j.out
#SBATCH --exclude=n3,n60,n76,n77,n80
exec 2>&1

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

mkdir -p /home1/doyoonkim/projects/elsa/logs
cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=c4 \
    --data_path="/home1/doyoonkim/projects/elsa/data/c4" \
    --nsamples=8 \
    --sparsity_ratio=0.5 \
    --sparsity_type=unstructured \
    --do_safe=true \
    --safe_lr=2e-4 \
    --safe_lmda=1e-3 \
    --safe_rho=0.05 \
    --safe_epochs=2 \
    --safe_warmup_epochs=1 \
    --safe_interval=4 \
    --safe_batch_size=2 \
    --safe_accumulation_steps=1 \
    --admm_beta1=0.9 \
    --admm_beta2=0.999 \
    --save_model=false \
    --eval_math500=false \
    --eval_zero_shot=false \
    --eval_full_bench=false \
    --wandb=false \
    --seed=42

echo "##### END #####"
