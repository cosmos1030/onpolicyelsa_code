#!/bin/bash
#SBATCH --job-name=elsa_math_prompt_t
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
########################################
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
DATA=/home1/doyoonkim/projects/elsa/data/math_220k_prompts.jsonl
SAVE_DIR=/home1/doyoonkim/projects/elsa/models

SPARSITY=0.5
ADMM_STEPS=4096
ADMM_LR=1e-2
ADMM_LMDA=5e-5
ADMM_LMDA_SCHEDULE=cosine
ADMM_INTERVAL=32
SEQLEN=2048
BS=2
ACCUM=1   # global batch = BS * ACCUM * 4GPU = 8

########################################
# Run
########################################
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home1/doyoonkim/miniconda3/envs/elsa/bin/accelerate launch --config_file config/default.yaml main.py \
  --model="${MODEL}" \
  --dataset=math_prompt \
  --data_path="${DATA}" \
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
  --admm_beta2=0.999 \
  --admm_interval=${ADMM_INTERVAL} \
  --admm_base_optimizer=adamw \
  --admm_precision=bf16 \
  --save_model=True \
  --admm_save_path="${SAVE_DIR}" \
  --eval_zero_shot=False \
  --wandb=True \
  --wandb_project=elsa_deepseek_1.5b \
  --seed=42

echo "##### END #####"