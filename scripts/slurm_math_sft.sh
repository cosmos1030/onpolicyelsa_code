#!/bin/bash
#SBATCH --job-name=r1_sparse_ft
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

########################################
# 2. Paths  (edit before submitting)
########################################
PRUNED_MODEL=/home1/doyoonkim/projects/RAC/open-r1-main/models/DeepSeek-R1-Distill-Qwen-1_pruned_50_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k          # pruned model checkpoint

DATASET_NAME=/home1/doyoonkim/projects/RAC/open-r1-main/math_trace/dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k_.jsonl

########################################
# 3. Sparse fine-tuning with GRPO
########################################
cd /home1/doyoonkim/projects/RAC/open-r1-main

torchrun --nproc_per_node=4 src/open_r1/sft.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/sft/config_demo.yaml \
  --model_name_or_path ${PRUNED_MODEL} \
  --dataset_name ${DATASET_NAME} \
  --dataset_text_field prompt \
  --do_train True \
  --do_eval False \
  --sparse_optimizer MaskedAdam \
  --learning_rate 2e-5 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --bf16 True \
  --gradient_checkpointing True \
  --report_to wandb \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --output_dir /home1/doyoonkim/projects/RAC/open-r1-main/models/sft/${PRUNED_MODEL##*/}_sft \
  --seed 42