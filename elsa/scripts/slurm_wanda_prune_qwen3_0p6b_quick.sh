#!/bin/bash
#SBATCH --job-name=wanda_qwen3_0p6b_quick
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/wanda_qwen3_0p6b_quick_%j.out
exec 2>&1

# Fast one-shot Wanda prune of Qwen3-0.6B to 30% sparsity, --smoketest to skip
# the expensive lighteval pass -- only need the pruned checkpoint itself,
# as a small/fast base for the saliency_snapshot_diagnostic pilot (avoids
# waiting on the slower ALPS 1.7B-s70 path, which needed 80GB and kept
# hitting queue congestion / OOM edge cases). No 0.5B exists in the Qwen3
# family (0.6B is the smallest); no ALPS-pruned checkpoint exists for 0.6B
# either, hence Wanda (one-shot, minutes not hours) instead.
#
# Usage: sbatch slurm_wanda_prune_qwen3_0p6b_quick.sh [SPARSITY]

SPARSITY=${1:-0.3}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
SAVE_PATH="/home1/doyoonkim/projects/elsa/models/qwen3_0p6b_wanda_s${SPARSITY_PCT}pct_quick"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "=== Wanda Qwen3-0.6B quick prune (s${SPARSITY_PCT}%) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "SAVE_PATH=$SAVE_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/RAC/open-r1-main

$PYTHON src/open_r1/prune_and_eval.py \
    --model_path "$MODEL" \
    --method wanda \
    --sparsity "$SPARSITY" \
    --nsamples 128 \
    --seqlen 2048 \
    --save_path "$SAVE_PATH" \
    --wandb_project reasoning_qwen3_0p6b_scratch \
    --wandb_name "wanda_quick_s${SPARSITY_PCT}" \
    --smoketest

echo "##### END #####"
