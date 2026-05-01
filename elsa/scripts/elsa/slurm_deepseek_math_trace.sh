#!/bin/bash
#SBATCH --job-name=elsa_deepseek
#SBATCH --partition=RTX4090
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=output/%j.out
#SBATCH -t 3-00:00:00

mkdir -p output

########################################
# 1. Conda
########################################
########################################
# 2. Paths & Hyperparameters
########################################
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
TRACE_DATA=/home1/doyoonkim/projects/RAC/open-r1-main/math_trace/dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k_.jsonl
SAVE_DIR=/home1/doyoonkim/projects/elsa/models

SPARSITY=0.5
ADMM_STEPS=2000
ADMM_LR=2e-4
ADMM_LMDA=0.01
ADMM_INTERVAL=32
SEQLEN=2048
BS=2
ACCUM=4

########################################
# 3. Run
########################################
cd /home1/doyoonkim/projects/elsa

export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home1/doyoonkim/miniconda3/envs/elsa/bin/accelerate launch --config_file config/default.yaml main.py \
  --model="${MODEL}" \
  --dataset=math_trace \
  --data_path="${TRACE_DATA}" \
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
  --wandb_project=RAC_elsa_baseline \
  --seed=42

echo "##### END #####"
