#!/bin/bash
#SBATCH --job-name=r1_sparse_ft
#SBATCH --partition=A100-40GB
#SBATCH --qos=hpgpu
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

########################################
# 2. Paths  (edit before submitting)
########################################
PRUNED_MODEL=/home1/doyoonkim/projects/RAC/open-r1-main/models/DeepSeek-R1-Distill-Qwen-1_pruned_50_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k         # pruned model checkpoint
OUTPUT_DIR=/home1/doyoonkim/projects/RAC/open-r1-main/models/grpo_8k  # training checkpoints
DATASET_NAME=open-r1/OpenR1-Math-220k

########################################
# 3. Sparse fine-tuning with GRPO
########################################
cd /home1/doyoonkim/projects/RAC/open-r1-main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch --config_file recipes/deepspeed_zero2_4gpu.yaml src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_grpo_retrain.yaml \
  --model_name_or_path ${PRUNED_MODEL} \
  --output_dir ${OUTPUT_DIR} \
  --dataset_name ${DATASET_NAME} \
  --dataset_prompt_column problem \
  --do_train True \
  --do_eval False \
  --sparse_optimizer MaskedAdam \
  --learning_rate 1e-6 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type constant \
  --warmup_ratio 0.0 \
  --max_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --bf16 True \
  --gradient_checkpointing True \
  --num_generations 4 \
  --temperature 0.7 \
  --beta 0.0 \
  --max_completion_length 8000 \
  --reward_funcs accuracy \
  --reward_weights 1.0 \
  --report_to wandb \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --use_vllm False \
  --score_completions True \
  --seed 42