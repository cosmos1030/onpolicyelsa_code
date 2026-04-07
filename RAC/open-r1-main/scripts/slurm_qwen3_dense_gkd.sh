#!/bin/bash
#SBATCH --job-name=qwen3_dense_gkd
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%j_gkd.out
#SBATCH -t 1-00:00:00

set -euo pipefail
mkdir -p logs

########################################
# 1. Conda
########################################
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

########################################
# 2. Hyperparameters
########################################
MODEL=/home1/doyoonkim/projects/RAC/open-r1-main/models/Qwen3-0_pruned_30_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__OpenR1-Math-220k
TEACHER_MODEL=Qwen/Qwen3-0.6B   # self-distillation (frozen teacher)

DATASET=open-r1/OpenR1-Math-220k
MAX_STEPS=30             # 220k 중 일부만 사용
SAVE_STEPS=5             # 체크포인트 저장 간격
BS=1
ACCUM=8                  # effective batch = 4 GPU × 1 × 8 = 32
LR=1e-5
TEMPERATURE=1.0
MAX_NEW_TOKENS=4096
LMBDA=1.0                # 1.0 = fully on-policy
BETA=1.0                 # 1.0 = reverse KL

STUDENT_TAG=$(basename ${MODEL})
OUTPUT_DIR=/home1/doyoonkim/projects/RAC/open-r1-main/models/${STUDENT_TAG}_gkd_lr${LR}_steps${MAX_STEPS}

WANDB_PROJECT=qwen3_gkd
WANDB_RUN_NAME=${STUDENT_TAG}_gkd_lr${LR}_steps${MAX_STEPS}

########################################
# 3. Env
########################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPEN_R1_SFT_CACHE_DIR=~/.cache/open-r1/sft_datasets/$(basename ${OUTPUT_DIR})
export WANDB_PROJECT
export WANDB_RUN_NAME

########################################
# 4. Train
########################################
cd /home1/doyoonkim/projects/RAC/open-r1-main

echo "=== Starting GKD training ==="
echo "Model: ${MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "Max steps: ${MAX_STEPS}, Save every: ${SAVE_STEPS}"

accelerate launch --config_file recipes/accelerate_1gpu.yaml src/open_r1/gkd.py \
  --model_name_or_path ${MODEL} \
  --teacher_model_name_or_path ${TEACHER_MODEL} \
  --output_dir ${OUTPUT_DIR} \
  --dataset_name ${DATASET} \
  --dataset_prompt_column problem \
  --system_prompt "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>" \
  --do_train True \
  --do_eval False \
  --lmbda ${LMBDA} \
  --beta ${BETA} \
  --temperature ${TEMPERATURE} \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --learning_rate ${LR} \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --kd_topk=50 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
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
  --save_total_limit 10 \
  --seed 42

echo "=== Training done. Checkpoints saved to ${OUTPUT_DIR} ==="
echo "=== MATH-500 eval runs in-process via MathEvalCallback on every save ==="

date
echo "##### END #####"
