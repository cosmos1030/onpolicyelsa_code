#!/bin/bash
#SBATCH --job-name=qwen3_trace
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs_trace/%j.out
#SBATCH -t 3-00:00:00

mkdir -p logs_trace

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/RAC/open-r1-main

python src/open_r1/grpo.py \
  --config recipes/Qwen3-0.6B/grpo/config_pruning.yaml \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_name open-r1/OpenR1-Math-220k \
  --dataset_prompt_column problem \
  --save_dir /home1/doyoonkim/projects/RAC/open-r1-main/math_trace \
  --num_generations 2 \
  --beta 0 \
  --report_to wandb \
  --use_vllm False \
  --max_completion_length 8192 \
  --do_train False \
  --trace_only \
  --trace_tokens 1000000

echo "##### END #####"
