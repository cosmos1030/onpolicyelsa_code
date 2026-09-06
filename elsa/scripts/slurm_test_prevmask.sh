#!/bin/bash
#SBATCH --job-name=test_prevmask
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
#SBATCH -t 0-01:30:00
#SBATCH --exclude=n3,n60,n80

exec 2>&1

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/wandb"

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_START_METHOD=thread

echo "Node: $(hostname)"

/home1/doyoonkim/miniconda3/envs/rac/bin/python main.py \
    --model="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562" \
    --dataset=mixed_cot \
    --data_path=/home1/doyoonkim/projects/elsa/data/ot3_fineweb_20k.jsonl \
    --gmp_prompt_path=/home1/doyoonkim/projects/elsa/data/ot3_fineweb_20k.jsonl \
    --do_gmp=true \
    --sparsity_ratio=0.5 \
    --gmp_steps=200 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=8 \
    --gmp_fisher_beta=0.999 \
    --gmp_kd_lambda=1.0 \
    --gmp_onpolicy_kd_lambda=1.0 \
    --gmp_opkd_use_vllm=true \
    --gmp_opkd_vllm_gpu_mem=0.35 \
    --gmp_opkd_prev_mask_teacher=true \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_max_seq_len=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_dpo_lambda=0.0 \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.005 \
    --gmp_tr_kl_threshold=0.01 \
    --gmp_tr_kl_reduce=mean \
    --save_model=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=false

echo "##### END #####"
