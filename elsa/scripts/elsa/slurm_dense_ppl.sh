#!/bin/bash
#SBATCH --job-name=dense_ppl
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=output/%j.out
#SBATCH -t 0-01:00:00

mkdir -p output

MODEL=Qwen/Qwen3-0.6B

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home1/doyoonkim/projects/elsa

/home1/doyoonkim/miniconda3/envs/rac/bin/python main.py \
  --model="${MODEL}" \
  --sparsity_ratio=0.0 \
  --sparsity_type=unstructured \
  --seqlen=2048 \
  --eval_zero_shot=True \
  --admm_steps=0 \
  --wandb=False \
  --seed=42

echo "##### END #####"
