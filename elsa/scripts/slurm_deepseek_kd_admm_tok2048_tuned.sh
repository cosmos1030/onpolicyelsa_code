#!/bin/bash
#SBATCH --job-name=elsa_qwen3_kd_2k
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=output_qwen/%j.out
#SBATCH -t 3-00:00:00

mkdir -p output_qwen

########################################
# Paths & Hyperparameters
########################################
MODEL=Qwen/Qwen3-0.6B
KD_DATA=/home1/doyoonkim/projects/elsa/data/math_220k_prompts.jsonl
SAVE_DIR=/home1/doyoonkim/projects/elsa/models

SPARSITY=0.5
ADMM_STEPS=1024
ADMM_LR=1e-2
ADMM_LMDA=5e-5
ADMM_LMDA_SCHEDULE=cosine
ADMM_INTERVAL=32
BS=1
ACCUM=8

KD_MAX_PROMPT_LEN=512
KD_MAX_NEW_TOKENS=2048
KD_TEMPERATURE=1.0

########################################
# Run
########################################
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home1/doyoonkim/miniconda3/envs/elsa/bin/python main.py \
  --model="${MODEL}" \
  --sparsity_ratio=${SPARSITY} \
  --sparsity_type=unstructured \
  --do_kd_admm=True \
  --kd_data_path="${KD_DATA}" \
  --kd_max_prompt_len=${KD_MAX_PROMPT_LEN} \
  --kd_max_new_tokens=${KD_MAX_NEW_TOKENS} \
  --kd_temperature=${KD_TEMPERATURE} \
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
  --wandb_project=elsa_qwen3_0.6b \
  --seed=42

echo "##### END #####"