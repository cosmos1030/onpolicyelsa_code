#!/bin/bash
#SBATCH --job-name=r1_gkd_220k
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
# 2. Paths & Hyperparameters
########################################
PRUNED_MODEL=/home1/doyoonkim/projects/RAC/open-r1-main/models/DeepSeek-R1-Distill-Qwen-1_pruned_50_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k
TEACHER_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
DATASET_NAME=open-r1/OpenR1-Math-220k

LR=1e-3
LMBDA=1.0
BETA=1.0
TEMPERATURE=1.0
MAX_NEW_TOKENS=4096
BS=1
ACCUM=4
WARMUP=0.0
LR_SCHEDULER=constant
MAX_STEPS=300
OPTIMIZER=ProjectedMuon  # MaskedAdam or adam or ProjectedMuon

OUTPUT_DIR=/home1/doyoonkim/projects/RAC/open-r1-main/models/gkd_220k_lr${LR}_lmb${LMBDA}_b${BETA}_t${TEMPERATURE}_tok${MAX_NEW_TOKENS}_bs${BS}_accum${ACCUM}_warm${WARMUP}_${LR_SCHEDULER}_steps${MAX_STEPS}_opt${OPTIMIZER}

########################################
# 3. On-policy distillation (GKD)
########################################
cd /home1/doyoonkim/projects/RAC/open-r1-main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPEN_R1_SFT_CACHE_DIR=~/.cache/open-r1/sft_datasets/${OUTPUT_DIR##*/}

accelerate launch --config_file recipes/deepspeed_zero2_4gpu.yaml src/open_r1/gkd.py \
  --model_name_or_path ${PRUNED_MODEL} \
  --teacher_model_name_or_path ${TEACHER_MODEL} \
  --output_dir ${OUTPUT_DIR} \
  --dataset_name ${DATASET_NAME} \
  --dataset_prompt_column problem \
  --do_train True \
  --do_eval False \
  --lmbda ${LMBDA} \
  --beta ${BETA} \
  --temperature ${TEMPERATURE} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --sparse_optimizer ${OPTIMIZER} \
  --learning_rate ${LR} \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type ${LR_SCHEDULER} \
  --warmup_ratio ${WARMUP} \
  --max_steps ${MAX_STEPS} \
  --per_device_train_batch_size ${BS} \
  --gradient_accumulation_steps ${ACCUM} \
  --bf16 True \
  --gradient_checkpointing True \
  --report_to wandb \
  --logging_steps 1 \
  --save_strategy no \
  --seed 42
