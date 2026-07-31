#!/bin/bash
#SBATCH --job-name=dbg_cubic_admm_0.6b
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/dbg_cubic_admm_0.6b_%j.out
exec 2>&1

# Debug: TR-z ADMM on Qwen3-0.6B, single RTX3090
# Verifies: TR-z iter logs appear, tr_z/ wandb metrics logged, binary search runs

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_cot_debug100.jsonl"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)

echo "=== DEBUG: TR-z ADMM Qwen3-0.6B RTX3090 ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

mkdir -p /home1/doyoonkim/projects/elsa/logs
cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=math_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=0.5 \
    --admm_steps=128 \
    --admm_interval=32 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=4 \
    --admm_lr=1e-5 \
    --admm_lmda=0.001 \
    --admm_lmda_schedule_mode=cosine \
    --admm_base_optimizer=adamw \
    --admm_beta2=0.999 \
    --admm_precision=bf16 \
    --admm_tr_z_proj=true \
    --admm_z_schedule_mode=cubic \
    --admm_cubic_steps=3 \
    --admm_tr_init_delta=0.05 \
    --admm_tr_delta_min=0.001 \
    --admm_tr_max_iters=8 \
    --kd_max_prompt_len=512 \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42

echo "=== EXIT: $? ==="
