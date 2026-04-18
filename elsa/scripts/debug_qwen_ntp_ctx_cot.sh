#!/bin/bash
#SBATCH --job-name=debug_ntp_ctx
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j_debug_ntp_ctx.out
#SBATCH -t 0-02:00:00
#SBATCH --exclude=n3

set -euo pipefail
echo "Node: $(hostname)"
echo "Start: $(date)"
source ~/miniconda3/etc/profile.d/conda.sh

cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader

/home1/doyoonkim/miniconda3/envs/rac/bin/python main.py \
  --model /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --dataset math_cot \
  --data_path /home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
  --sparsity_ratio 0.3 \
  --admm_steps 40 \
  --admm_batch_size 1 \
  --admm_gradient_accumulation_steps 8 \
  --admm_lmda 0.01 \
  --admm_lr 1e-4 \
  --admm_interval 32 \
  --admm_precision bf16 \
  --kd_max_prompt_len 512 \
  --admm_logging_steps 1 \
  --admm_eval_steps 10 \
  --nosave_model \
  --noeval_zero_shot \
  --eval_math500 \
  --math500_max_new_tokens 2048 \
  --wandb \
  --wandb_project elsa_qwen3_0.6b \
  --seed 42

echo "=== Done ==="
echo "End: $(date)"
