#!/bin/bash
#SBATCH --job-name=sparsegpt_ot_fw
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --exclude=n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/sparsegpt_ot_fw_%j.out
exec 2>&1

# SparseGPT pruning with OpenThoughts3 80% + FineWeb-Edu 20% calibration.
# Matches ReasoningQAT paper calibration setup.
#
# Usage:
#   sbatch slurm_sparsegpt_ot_fineweb.sh <MODEL_PATH> <SPARSITY> [SAVE_PATH]
#
# Examples:
#   # Unstructured 50%
#   sbatch slurm_sparsegpt_ot_fineweb.sh /path/to/model 0.5
#
#   # 2:4 structured
#   sbatch slurm_sparsegpt_ot_fineweb.sh /path/to/model 0.5 /path/to/save --prune_n 2 --prune_m 4

MODEL_PATH=${1:?"Usage: sbatch slurm_sparsegpt_ot_fineweb.sh <MODEL_PATH> <SPARSITY> [SAVE_PATH]"}
SPARSITY=${2:-0.5}
SAVE_PATH=${3:-""}

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

# Default save path: same dir as model, with sparsity suffix
if [ -z "$SAVE_PATH" ]; then
    SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
    SAVE_PATH="${MODEL_PATH}_sgpt_s${SPARSITY_PCT}pct_ot_fw"
fi

echo "=== SparseGPT pruning ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "SPARSITY=$SPARSITY"
echo "SAVE_PATH=$SAVE_PATH"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

cd /home1/doyoonkim/projects/RAC/open-r1-main

$PYTHON src/open_r1/prune_sparsegpt.py \
    --model_path "$MODEL_PATH" \
    --sparsity "$SPARSITY" \
    --nsamples 128 \
    --seqlen 2048 \
    --seed 42 \
    --save_path "$SAVE_PATH"

echo "=== Done ==="
date
