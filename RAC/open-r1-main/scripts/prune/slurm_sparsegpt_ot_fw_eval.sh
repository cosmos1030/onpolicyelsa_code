#!/bin/bash
#SBATCH --job-name=sgpt_ot_fw_eval
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --exclude=n52,n55,n58,n80
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs/sgpt_ot_fw_eval_%j.out
exec 2>&1

# SparseGPT prune + full eval (PPL, zero-shot, 5 lighteval benchmarks).
# Calibration: OpenThoughts3 80% + FineWeb-Edu 20%.
# Results logged to wandb project: reasoning_pruning_v1.
#
# Usage:
#   sbatch slurm_sparsegpt_ot_fw_eval.sh <MODEL_PATH> <SPARSITY> <NSAMPLES> [SAVE_PATH] [WANDB_NAME]
#
# Smoke test (fast pipeline check):
#   sbatch slurm_sparsegpt_ot_fw_eval.sh <MODEL_PATH> 0.5 4 /tmp/sgpt_smoketest "" --smoketest

MODEL_PATH=${1:?"Usage: sbatch slurm_sparsegpt_ot_fw_eval.sh <MODEL_PATH> <SPARSITY> <NSAMPLES> [SAVE_PATH] [WANDB_NAME] [--smoketest]"}
SPARSITY=${2:-0.5}
NSAMPLES=${3:-128}
SAVE_PATH=${4:-""}
WANDB_NAME=${5:-""}
SMOKETEST=${6:-""}
EXTRA_ARGS=${7:-""}  # e.g. --skip_prune

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb

if [ -z "$SAVE_PATH" ]; then
    SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
    MODEL_BASE=$(basename "$MODEL_PATH")
    SAVE_PATH="/local-data/user-data/$USER/job_$SLURM_JOB_ID/${MODEL_BASE}_sgpt_s${SPARSITY_PCT}pct_n${NSAMPLES}_ot_fw"
fi

export WANDB_DIR=/local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb
export WANDB_START_METHOD=thread
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1 | tr -d ' \n\r')
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

echo "=== SparseGPT + Eval Pipeline ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "SPARSITY=$SPARSITY"
echo "NSAMPLES=$NSAMPLES"
echo "SAVE_PATH=$SAVE_PATH"
echo "WANDB_NAME=$WANDB_NAME"
echo "SMOKETEST=$SMOKETEST"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date

cd /home1/doyoonkim/projects/RAC/open-r1-main

ARGS=(
    --model_path "$MODEL_PATH"
    --sparsity "$SPARSITY"
    --nsamples "$NSAMPLES"
    --seqlen 2048
    --save_path "$SAVE_PATH"
    --wandb_project reasoning_pruning_v1
)

[ -n "$WANDB_NAME" ] && ARGS+=(--wandb_name "$WANDB_NAME")
[ "$SMOKETEST" = "--smoketest" ] && ARGS+=(--smoketest)
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

$PYTHON src/open_r1/prune_and_eval.py "${ARGS[@]}"

echo "=== Done ==="
date
