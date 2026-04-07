#!/bin/bash
#SBATCH --job-name=elsa_deepseek_c4
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=output/%j.out
#SBATCH -t 3-00:00:00

mkdir -p output

########################################
# 1. Paths & Hyperparameters
########################################
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
SAVE_DIR=/home1/doyoonkim/projects/elsa/models

SPARSITY=0.5
ADMM_STEPS=4096
ADMM_LR=2e-4
ADMM_LMDA=0.01
ADMM_INTERVAL=32
SEQLEN=2048
BS=2
ACCUM=4

########################################
# 2. Pre-download C4 (single process to avoid multi-rank conflicts)
########################################
cd /home1/doyoonkim/projects/elsa

# Clear incomplete cache if exists
rm -f ~/.cache/huggingface/datasets/allenai___c4/*/0.0.0/*.incomplete_info.lock 2>/dev/null || true

/home1/doyoonkim/miniconda3/envs/elsa/bin/python -c "
from datasets import load_dataset
print('Pre-downloading C4 train split...')
ds = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', trust_remote_code=True, cache_dir='~/.cache/huggingface/datasets')
print(f'C4 train ready: {len(ds)} samples')
print('Pre-downloading C4 validation split...')
ds = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', trust_remote_code=True, cache_dir='~/.cache/huggingface/datasets')
print(f'C4 validation ready: {len(ds)} samples')
"

########################################
# 3. Run
########################################
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
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
  --admm_interval=${ADMM_INTERVAL} \
  --admm_base_optimizer=adamw \
  --admm_precision=bf16 \
  --save_model=True \
  --admm_save_path="${SAVE_DIR}" \
  --admm_save_inputs=True \
  --eval_zero_shot=False \
  --wandb=True \
  --wandb_project=elsa_deepseek_1.5b \
  --seed=42

echo "##### END #####"
