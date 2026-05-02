#!/bin/bash
#SBATCH --job-name=eval_dense_ds1.5b
#SBATCH --partition=RTX4090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
#SBATCH -t 0-00:08:00
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

/home1/doyoonkim/miniconda3/envs/rac/bin/python main.py \
    --model=$MODEL \
    --dataset=math_cot \
    --data_path=/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
    --sparsity_ratio=0 \
    --eval_math500=true \
    --math500_max_new_tokens=8192 \
    --math500_max_samples=100 \
    --math500_model_path=$MODEL \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=gmp_qwen3_1.5b \
    --seed=42 \
    --push_to_hub=false \
    --save_model=false

echo "##### END #####"
