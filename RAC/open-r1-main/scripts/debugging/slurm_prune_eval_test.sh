#!/bin/bash
#SBATCH --job-name=prune_eval_TEST
#SBATCH --partition=A6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=n44
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%j_prune_eval_test.out
#SBATCH -t 0-01:00:00

set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

# Minimal pruning + post-prune MATH-500 eval test.
# Small calibration token budget + tiny eval set for fast verification.
cd /home1/doyoonkim/projects/RAC/open-r1-main

export WANDB_PROJECT=prune_eval_test
export WANDB_NAME=prune_eval_callback_test
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon

python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name /home1/doyoonkim/projects/RAC/open-r1-main/math_trace/dataset_Qwen3-0_trace_OpenR1-Math-220k \
  --dataset_prompt_column problem \
  --beta 0 \
  --report_to wandb \
  --use_vllm False \
  --do_train False \
  --prune \
  --pruning_method SparseGPT \
  --prune_sparsity 0.3 \
  --prune_calib_tokens 20000 \
  --push_to_hub False \
  --score_completions False \
  --math_eval_samples 5 \
  --math_eval_max_new_tokens 512 \
  --math_eval_batch_size 4

echo "=== Test done ==="
date