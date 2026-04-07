#!/bin/bash
#SBATCH --job-name=debug_ntp_cot
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j.out
#SBATCH -t 0-01:00:00

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL=/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca

$PYTHON main.py \
    --model=$MODEL \
    --dataset=math_cot \
    --data_path=/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
    --sparsity_ratio=0.5 \
    --admm_steps=50 \
    --admm_batch_size=8 \
    --admm_gradient_accumulation_steps=1 \
    --admm_interval=10 \
    --admm_lr=0.01 \
    --admm_lmda=5e-5 \
    --admm_lmda_schedule_mode=cosine \
    --admm_beta2=0.999 \
    --admm_base_optimizer=adamw \
    --admm_precision=bf16 \
    --save_model=false \
    --eval_math500=false \
    --eval_zero_shot=false \
    --wandb_project=elsa_debug \
    --seed=42

echo "##### END #####"