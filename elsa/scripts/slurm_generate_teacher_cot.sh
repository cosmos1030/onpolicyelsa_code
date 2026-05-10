#!/bin/bash
#SBATCH --job-name=gen_teacher_cot
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
#SBATCH -t 0-08:00:00

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE"

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
INPUT="/home1/doyoonkim/projects/elsa/data/math_220k_prompts.jsonl"
OUTPUT="/home1/doyoonkim/projects/elsa/data/math_20k_teacher_cot.jsonl"

echo "Node: $(hostname)"

/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/generate_teacher_cot.py \
    --model=$MODEL \
    --input=$INPUT \
    --output=$OUTPUT \
    --max_samples=20000 \
    --max_new_tokens=8192 \
    --temperature=0.6 \
    --top_p=0.95 \
    --tensor_parallel_size=1

echo "##### END #####"
