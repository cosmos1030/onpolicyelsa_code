#!/bin/bash
#SBATCH --job-name=gmp_${METHOD}_s${SPARSITY_PCT}
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
#SBATCH -t 1-00:00:00
#SBATCH --exclude=n3,n76,n80

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
export WANDB_DIR="$LOCAL_JOB_BASE/wandb"

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

echo "Node: $(hostname)"
echo "LOCAL_JOB_BASE: $LOCAL_JOB_BASE"
echo "METHOD: ${METHOD}, SPARSITY: ${SPARSITY}, KD_LAMBDA: ${KD_LAMBDA}"

/home1/doyoonkim/miniconda3/envs/rac/bin/python main.py \
    --model=$MODEL \
    --dataset=math_cot \
    --data_path=/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
    --sparsity_ratio=${SPARSITY} \
    --do_gmp=true \
    --gmp_steps=1024 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --gmp_lr=${LR} \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=32 \
    --gmp_fisher_beta=0.999 \
    --gmp_kd_lambda=${KD_LAMBDA} \
    --gmp_kd_temperature=2.0 \
    --gmp_kd_topk=100 \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --gmp_max_prompt_len=512 \
    --gmp_max_seq_len=2048 \
    --save_model=true \
    --eval_math500=true \
    --math500_max_new_tokens=8192 \
    --math500_max_samples=100 \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=gmp_qwen3_1.5b \
    --seed=42 \
    --push_to_hub=true

echo "##### END #####"
