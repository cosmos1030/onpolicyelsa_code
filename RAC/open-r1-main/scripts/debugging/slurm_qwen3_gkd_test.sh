#!/bin/bash
#SBATCH --job-name=qwen3_gkd_TEST
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j_gkd_test.out
#SBATCH -t 0-02:00:00

set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

# Minimal test: 2 train steps, save every step → eval fires twice.
# MATH-500 restricted to 10 samples, short max_new_tokens to keep it fast.
MODEL=/home1/doyoonkim/projects/RAC/open-r1-main/models/Qwen3-0_pruned_30_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__OpenR1-Math-220k
TEACHER_MODEL=Qwen/Qwen3-0.6B
DATASET=open-r1/OpenR1-Math-220k

MAX_STEPS=2
SAVE_STEPS=1
BS=1
ACCUM=2
LR=1e-5
MAX_NEW_TOKENS=512

STUDENT_TAG=$(basename ${MODEL})
OUTPUT_DIR=/home1/doyoonkim/projects/RAC/open-r1-main/models/TEST_${STUDENT_TAG}_gkd_cbtest

WANDB_PROJECT=qwen3_gkd_test
WANDB_RUN_NAME=gkd_callback_test

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPEN_R1_SFT_CACHE_DIR=~/.cache/open-r1/sft_datasets/test
export WANDB_PROJECT
export WANDB_RUN_NAME

cd /home1/doyoonkim/projects/RAC/open-r1-main

# remove any stale output dir so resume doesn't skip training
rm -rf "${OUTPUT_DIR}"

echo "=== Test GKD + MathEvalCallback ==="
accelerate launch --config_file recipes/accelerate_1gpu.yaml src/open_r1/gkd.py \
  --model_name_or_path ${MODEL} \
  --teacher_model_name_or_path ${TEACHER_MODEL} \
  --output_dir ${OUTPUT_DIR} \
  --dataset_name ${DATASET} \
  --dataset_prompt_column problem \
  --system_prompt "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>" \
  --do_train True \
  --do_eval False \
  --lmbda 1.0 \
  --beta 1.0 \
  --temperature 1.0 \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --learning_rate ${LR} \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --kd_topk=50 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.0 \
  --max_steps ${MAX_STEPS} \
  --per_device_train_batch_size ${BS} \
  --gradient_accumulation_steps ${ACCUM} \
  --bf16 True \
  --gradient_checkpointing True \
  --torch_dtype bfloat16 \
  --attn_implementation flash_attention_2 \
  --run_name ${WANDB_RUN_NAME} \
  --report_to wandb \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit 2 \
  --math_eval_samples 10 \
  --math_eval_max_new_tokens 512 \
  --math_eval_batch_size 4 \
  --seed 42

echo "=== Test done ==="
date