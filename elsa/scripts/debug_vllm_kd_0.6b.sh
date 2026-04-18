#!/bin/bash
#SBATCH --job-name=vllm_kd_0.6b
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j_vllm_kd_0.6b.out
#SBATCH -t 0-01:00:00
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
  --do_kd_admm true \
  --kd_data_path /home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
  --kd_use_cot_dataset true \
  --kd_lambda 0.5 \
  --kd_max_new_tokens 512 \
  --kd_max_prompt_len 512 \
  --kd_nsamples 200 \
  --kd_topk 50 \
  --kd_temperature 1.0 \
  --kd_use_vllm true \
  --kd_vllm_gpu_memory_utilization 0.15 \
  --kd_buffer_size 32 \
  --kd_buffer_refresh_interval 32 \
  --admm_logging_steps 1 \
  --nosave_model \
  --noeval_zero_shot \
  --eval_math500 \
  --math500_max_new_tokens 2048 \
  --nowandb \
  --seed 42

echo "=== Done ==="
echo "End: $(date)"
