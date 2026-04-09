#!/bin/bash
#SBATCH --job-name=prune_eval_debug
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs/%j_debug.out
#SBATCH -t 0-01:00:00

set -euo pipefail
mkdir -p /home1/doyoonkim/projects/RAC/open-r1-main/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/RAC/open-r1-main

export WANDB_PROJECT=prune_eval_debug
export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon

/home1/doyoonkim/miniconda3/envs/rac/bin/python src/open_r1/grpo.py \
  --config recipes/Qwen3-0.6B/grpo/config_pruning.yaml \
  --model_name_or_path /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --dataset_name /home1/doyoonkim/projects/RAC/open-r1-main/math_trace/dataset_Qwen3-0_trace_OpenR1-Math-220k \
  --do_train False \
  --prune \
  --pruning_method SparseGPT \
  --prune_sparsity 0.3 \
  --prune_calib_tokens 20000 \
  --push_to_hub False \
  --score_completions False \
  --report_to wandb

echo "=== Test done ==="
date