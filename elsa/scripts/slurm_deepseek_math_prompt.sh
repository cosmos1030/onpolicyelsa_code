#!/bin/bash
#SBATCH --job-name=elsa_math_prompt
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
########################################
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
DATA=/home1/doyoonkim/projects/elsa/data/math_220k_prompts.jsonl
SAVE_DIR=/home1/doyoonkim/projects/elsa/models

SPARSITY=0.5
ADMM_STEPS=4096
ADMM_LR=2e-4
ADMM_LMDA=0.01
ADMM_INTERVAL=32
SEQLEN=2048
BS=2
ACCUM=16

########################################
# Run
########################################
export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home1/doyoonkim/miniconda3/envs/elsa/bin/python main.py \
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
