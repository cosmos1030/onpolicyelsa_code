#!/bin/bash
#SBATCH --job-name=offpolicy_kd_0.6b
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j_offpolicy_kd_0.6b.out
#SBATCH -t 0-00:30:00
#SBATCH --exclude=n3,n80

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
  --seqlen 1024 \
  --sparsity_ratio 0.3 \
  --admm_steps 40 \
  --admm_batch_size 1 \
  --admm_gradient_accumulation_steps 8 \
  --admm_lmda 0.005 \
  --admm_lr 5e-5 \
  --admm_lmda_schedule_mode cosine \
  --admm_beta2 0.999 \
  --admm_interval 32 \
  --admm_precision bf16 \
  --do_offpolicy_kd_admm true \
  --kd_data_path /home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
  --kd_nsamples 200 \
  --kd_lambda 0.05 \
  --kd_max_prompt_len 512 \
  --kd_topk 50 \
  --kd_temperature 1.0 \
  --admm_logging_steps 1 \
  --nosave_model \
  --noeval_zero_shot \
  --noeval_math500 \
  --nowandb \
  --seed 42

echo "=== Done ==="
echo "End: $(date)"
