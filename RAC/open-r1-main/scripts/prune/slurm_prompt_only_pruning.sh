#!/bin/bash
#SBATCH --job-name=prune
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/%j.out
#SBATCH -t 3-00:00:00

mkdir -p logs

########################################
# 1. Conda
########################################
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac


python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --dataset_name open-r1/OpenR1-Math-220k \
  --dataset_prompt_column problem \
  --beta 0 \
  --report_to wandb \
  --use_vllm False \
  --do_train False \
  --prune \
  --pruning_method SparseGPT \
  --prune_sparsity 0.4 \
  --prune_calib_tokens 1_000_000 \
  --push_to_hub False \
  --score_completions False