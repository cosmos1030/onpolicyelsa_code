#!/bin/bash
#SBATCH --job-name=debug_grpo2
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=0-00:30:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/debug_grpo_opkd2_%j.out

exec 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_MODE=disabled

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

echo "Node: $(hostname)"
echo "Testing with sweep-equivalent params (max_new_tokens=8192, num_rollouts=8)..."

/home1/doyoonkim/miniconda3/envs/rac/bin/python -u main.py \
  --model="$MODEL" \
  --dataset=math_cot \
  --data_path=/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
  --gmp_prompt_path=/home1/doyoonkim/projects/elsa/data/math_220k_with_answers.jsonl \
  --sparsity_ratio=0.5 \
  --do_grpo_opkd=true \
  --gmp_correctness_reward=true \
  --gmp_format_reward=0.1 \
  --gmp_lr=1e-4 \
  --gmp_steps=10 \
  --gmp_batch_size=1 \
  --gmp_grad_accum=8 \
  --gmp_warmup_ratio=0.05 \
  --gmp_mask_interval=8 \
  --gmp_fisher_beta=0.999 \
  --gmp_grpo_num_rollouts=8 \
  --gmp_grpo_interval=8 \
  --gmp_grpo_lambda=0.5 \
  --gmp_grpo_eps_clip=0.2 \
  --gmp_onpolicy_kd_lambda=0.0 \
  --gmp_onpolicy_mixed_alpha=0.2 \
  --gmp_onpolicy_max_new_tokens=8192 \
  --gmp_onpolicy_temperature=0.6 \
  --gmp_onpolicy_kd_topk=100 \
  --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
  --gmp_max_prompt_len=512 \
  --gmp_max_seq_len=2048 \
  --save_model=false \
  --eval_math500=false \
  --eval_zero_shot=false \
  --wandb=false \
  --seed=42 \
  --seqlen=2048

echo "##### END #####"
