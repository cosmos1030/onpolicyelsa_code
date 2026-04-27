#!/bin/bash
#SBATCH --job-name=dbg_ntp_mom30
#SBATCH --partition=3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/doyoonkim/onpolicyelsa_code/output_slurm/%j.out
#SBATCH -t 0-01:30:00

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh 2>/dev/null || true
cd /home/doyoonkim/onpolicyelsa_code/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/home/doyoonkim/.conda/envs/rac/lib/glibc_compat.so
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

PYTHON=/home/doyoonkim/.conda/envs/rac/bin/python
MODEL=/home/shared/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
DATA=/home/shared/dataset/math_cot_debug_500.jsonl

echo "=== DEBUG: NTP-CoT momentum 30pct ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "======================================"

$PYTHON main.py \
    --model=$MODEL \
    --dataset=math_cot \
    --data_path=$DATA \
    --sparsity_ratio=0.30 \
    --admm_steps=64 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=8 \
    --admm_interval=32 \
    --admm_lr=1.0e-5 \
    --admm_lmda=5.0e-3 \
    --admm_lmda_schedule_mode=cosine \
    --admm_beta2=0.999 \
    --admm_base_optimizer=adamw \
    --admm_precision=bf16 \
    --admm_projection_mode=momentum \
    --admm_logging_steps=8 \
    --save_model=false \
    --eval_math500=true \
    --math500_max_new_tokens=4096 \
    --eval_zero_shot=false \
    --wandb=false \
    --seed=42

echo "##### END #####"
date
