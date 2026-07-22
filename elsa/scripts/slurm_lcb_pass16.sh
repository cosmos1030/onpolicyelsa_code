#!/bin/bash
#SBATCH --job-name=lcb_pass16
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/%x_%j.out
#SBATCH -t 0-06:00:00
#SBATCH --exclude=n3,n51,n52,n55,n58,n60,n76,n80

# Usage:
#   sbatch scripts/slurm_lcb_pass16.sh deepseek-1.5b
#   sbatch scripts/slurm_lcb_pass16.sh qwen3-1.7b

MODEL=${1:-"deepseek-1.5b"}

OUT_BASE="/home1/doyoonkim/projects/elsa/eval_outputs/lcb_pass16"
mkdir -p "$OUT_BASE"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

OUT_PATH="$OUT_BASE/lcb_pass16_${MODEL}.json"

echo "Node: $(hostname), MODEL: $MODEL"
echo "Output: $OUT_PATH"

/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/eval_lcb_pass16.py \
    --model "$MODEL" \
    --n 16 \
    --temperature 0.6 \
    --max_tokens 8192 \
    --gpu_util 0.85 \
    --out "$OUT_PATH" \
    "${@:2}"

echo "##### END #####"
