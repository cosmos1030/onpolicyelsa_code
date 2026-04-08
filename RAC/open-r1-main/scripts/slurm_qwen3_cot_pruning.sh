#!/bin/bash
#SBATCH --job-name=prune_qwen3
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


/home1/doyoonkim/miniconda3/envs/rac/bin/python src/open_r1/grpo.py \
  --config recipes/Qwen3-0.6B/grpo/config_pruning.yaml \
  --model_name_or_path /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --dataset_name /home1/doyoonkim/projects/RAC/open-r1-main/math_trace/dataset_Qwen3-0_trace_OpenR1-Math-220k \
  --beta 0 \
  --report_to wandb \
  --use_vllm False \
  --do_train False \
  --prune \
  --pruning_method SparseGPT \
  --prune_sparsity 0.3 \
  --prune_calib_tokens 1_000_000 \
  --push_to_hub False \
  --score_completions False
