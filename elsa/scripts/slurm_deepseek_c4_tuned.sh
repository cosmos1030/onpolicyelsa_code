#!/bin/bash
#SBATCH --job-name=elsa_c4_tuned
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
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
ADMM_LR=1e-1
ADMM_LMDA=5e-5
ADMM_LMDA_SCHEDULE=cosine
ADMM_INTERVAL=32
SEQLEN=2048
BS=8
ACCUM=1   # global batch = 8 (paper setting)

########################################
# Run
########################################
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home1/doyoonkim/miniconda3/envs/elsa/bin/python main.py \
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
