#!/bin/bash
#SBATCH --job-name=vllm_kd_1.5b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j_vllm_kd_1.5b.out
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

# GPU 메모리 확인
nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader

/home1/doyoonkim/miniconda3/envs/rac/bin/python main.py \
  --model /home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
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
  --kd_interval 16 \
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
  --nowandb \
  --seed 42

echo "=== Done ==="
echo "End: $(date)"
