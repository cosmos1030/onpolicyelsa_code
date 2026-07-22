#!/bin/bash
#SBATCH --job-name=build_chosen_cache
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --exclude=n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/build_chosen_cache_%j.out
exec 2>&1

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_unified.jsonl"

echo "=== BUILD CHOSEN CACHE ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

# --- debug: 20 steps, batch_size=1, grad_accum=4 → 80 pairs ---
/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/build_chosen_cache.py \
    --model_path=$MODEL \
    --data_path=$DATA_PATH \
    --total_steps=20 \
    --batch_size=1 \
    --grad_accum=4 \
    --max_new_tokens=1024 \
    --temperature=0.7 \
    --max_prompt_len=512 \
    --cache_dir=/home1/doyoonkim/projects/elsa/.cache/dpo_chosen \
    --gen_batch_size=64 \
    --gpu_mem=0.85

echo "=== DONE ==="
