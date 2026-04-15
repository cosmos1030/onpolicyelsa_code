#!/bin/bash
#SBATCH --job-name=vllm_kd_test
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j_vllm_kd_test.out
#SBATCH -t 0-01:00:00

set -euo pipefail
echo "Node: $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/elsa

/home1/doyoonkim/miniconda3/envs/rac/bin/python main.py \
  --model /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --dataset math_cot \
  --data_path /home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
  --sparsity_ratio 0.3 \
  --admm_steps 10 \
  --admm_batch_size 1 \
  --admm_gradient_accumulation_steps 1 \
  --admm_lmda 0.01 \
  --admm_lr 1e-4 \
  --admm_precision bf16 \
  --do_kd_admm true \
  --kd_data_path /home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
  --kd_use_cot_dataset true \
  --kd_interval 2 \
  --kd_lambda 0.5 \
  --kd_max_new_tokens 32 \
  --kd_max_prompt_len 128 \
  --kd_nsamples 200 \
  --kd_topk 50 \
  --kd_temperature 1.0 \
  --kd_use_vllm true \
  --kd_vllm_gpu_memory_utilization 0.3 \
  --save_model false \
  --eval_zero_shot false \
  --eval_math500 false \
  --wandb false \
  --seed 42

echo "=== Done ==="
date
