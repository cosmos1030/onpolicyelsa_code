#!/bin/bash
#SBATCH --job-name=elsa_multigpu_test
#SBATCH --partition=A6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=output/%j.out
#SBATCH -t 0-00:30:00

mkdir -p output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

torchrun --nproc_per_node=2 --standalone main.py \
  --model=Qwen/Qwen3-0.6B \
  --dataset=math_prompt \
  --data_path=/home1/doyoonkim/projects/elsa/data/math_220k_prompts.jsonl \
  --sparsity_ratio=0.3 \
  --sparsity_type=unstructured \
  --seqlen=512 \
  --admm_steps=10 \
  --admm_batch_size=1 \
  --admm_gradient_accumulation_steps=2 \
  --admm_lr=1e-5 \
  --admm_lmda=1e-3 \
  --admm_interval=5 \
  --admm_precision=bf16 \
  --wandb=False \
  --save_model=False \
  --eval_zero_shot=False

echo "##### END #####"
