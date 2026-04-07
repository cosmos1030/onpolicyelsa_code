#!/bin/bash
#SBATCH --job-name=r1_trace
#SBATCH --partition=4A100
#SBATCH --qos=4A100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/%j.out
#SBATCH -t 3-00:00:00

python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
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