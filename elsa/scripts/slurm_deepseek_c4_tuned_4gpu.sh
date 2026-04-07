#!/bin/bash
#SBATCH --job-name=elsa_c4_tuned_4g
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=output/%j.out
#SBATCH -t 3-00:00:00

mkdir -p output

########################################
# Paths & Hyperparameters
# Based on paper Table 5: ~1.3B model, 50% sparsity
#   lr=1e-1, lmda=5e-5, schedule=cosine
########################################
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
SAVE_DIR=/home1/doyoonkim/projects/elsa/models

SPARSITY=0.5
ADMM_STEPS=4096
ADMM_LR=1e-2
ADMM_LMDA=5e-5
ADMM_LMDA_SCHEDULE=cosine
ADMM_INTERVAL=32
SEQLEN=2048
BS=2
ACCUM=1   # global batch = BS * ACCUM * 4GPU = 8 (paper setting)

########################################
# Run
########################################
cd /home1/doyoonkim/projects/elsa

# Clear incomplete C4 cache locks
rm -f ~/.cache/huggingface/datasets/allenai___c4/*/0.0.0/*.incomplete_info.lock 2>/dev/null || true

export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home1/doyoonkim/miniconda3/envs/elsa/bin/accelerate launch --config_file config/default.yaml main.py \
  --model="${MODEL}" \
  --dataset=c4 \
  --seqlen=${SEQLEN} \
  --sparsity_ratio=${SPARSITY} \
  --sparsity_type=unstructured \
  --admm_steps=${ADMM_STEPS} \
  --admm_batch_size=${BS} \
  --admm_gradient_accumulation_steps=${ACCUM} \
  --admm_lr=${ADMM_LR} \
  --admm_lmda=${ADMM_LMDA} \
  --admm_final_lmda=${ADMM_LMDA} \
  --admm_lmda_schedule_mode=${ADMM_LMDA_SCHEDULE} \
  --admm_interval=${ADMM_INTERVAL} \
  --admm_beta2=0.999 \
  --admm_base_optimizer=adamw \
  --admm_precision=bf16 \
  --save_model=True \
  --admm_save_path="${SAVE_DIR}" \
  --eval_zero_shot=False \
  --wandb=True \
  --wandb_project=elsa_deepseek_1.5b \
  --seed=42

echo "##### END #####"
