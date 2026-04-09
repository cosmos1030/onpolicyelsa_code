#!/bin/bash
#SBATCH --job-name=debug_math500
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
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
    --dataset=math_trace \
    --data_path=/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
    --sparsity_ratio=0.3 \
    --nsamples=32 \
    --admm_steps=5 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=4 \
    --admm_interval=5 \
    --admm_lr=1e-5 \
    --admm_lmda=5e-3 \
    --admm_lmda_schedule_mode=cosine \
    --admm_beta2=0.999 \
    --admm_base_optimizer=adamw \
    --admm_precision=bf16 \
    --admm_logging_steps=1 \
    --admm_num_train_samples=2 \
    --admm_num_eval_samples=2 \
    --save_model=true \
    --admm_save_path=/home1/doyoonkim/projects/elsa/models \
    --eval_math500=true \
    --math500_max_new_tokens=1024 \
    --eval_zero_shot=false \
    --wandb=false \
    --seed=42

echo "##### END #####"
date
