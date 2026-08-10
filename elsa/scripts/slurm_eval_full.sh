#!/bin/bash
#SBATCH --job-name=eval_full
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n76,n77,n80,n87,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_full_%j.out
exec 2>&1

# Full eval (PPL + zero-shot + lighteval bench, 7 tasks by default) for an
# existing checkpoint. Optional trailing args let you narrow this down to a
# lighteval-only rerun (skip PPL/zero-shot, pick specific benchmarks, set
# seed/tensor-parallel) -- useful for re-checking a benchmark after a config
# change, or for multi-seed variance runs.
#
# Current lighteval settings (lib/lighteval_bench.py, as of 2026-08-10):
#   math500/gpqa/ifeval/lcb: max_new_tokens=max_model_length=32768
#   aime24/aime25:           max_new_tokens=max_model_length=38912
#   gsm8k:                   2048/4096 (not in Qwen3's official post-training
#                            thinking-mode suite, no official budget to match)
#   sampling: temperature=0.6, top_p=0.95, top_k=20 (Qwen3 thinking-mode recipe)
#   Seeds: math500/ifeval/gsm8k/lcb are fine at 1 seed given their sample
#   counts; AIME24/25 (30 problems each) and GPQA (198 problems) need 3 seeds
#   (e.g. 42/0/1) -- 1 problem is a multi-point swing on AIME specifically.
#
# Usage: sbatch slurm_eval_full.sh <MODEL_PATH> <RUN_NAME> <METHOD> <SPARSITY> <WANDB_PROJECT> [BENCHMARKS] [SEED] [TP_SIZE] [SKIP_PPL] [SKIP_ZEROSHOT]
# e.g. (full eval, unchanged from before):
#   sbatch slurm_eval_full.sh /path/to/model sgpt_s60 sparsegpt 0.6 reasoning_qwen3_4b
# e.g. (AIME24/25 only, seed 0, tensor_parallel=2 -- pass GPU count via --gres too):
#   sbatch --gres=gpu:2 --cpus-per-task=16 --mem=96G slurm_eval_full.sh \
#     /path/to/Qwen3-8B aime_seed0_8b dense 0.0 reasoning_qwen3_8b aime24,aime25 0 2 true true

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_full.sh <MODEL_PATH> <RUN_NAME> <METHOD> <SPARSITY> <WANDB_PROJECT> [BENCHMARKS] [SEED] [TP_SIZE] [SKIP_PPL] [SKIP_ZEROSHOT]"}
RUN_NAME=${2:-"eval"}
METHOD=${3:-"sparsegpt"}
SPARSITY=${4:-0.0}
WANDB_PROJECT=${5:-"reasoning_pruning_v2"}
BENCHMARKS=${6:-}
SEED=${7:-42}
TP_SIZE=${8:-1}
SKIP_PPL=${9:-false}
SKIP_ZEROSHOT=${10:-false}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="/home1/doyoonkim/projects/elsa/logs/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY=$(grep "^export WANDB_API_KEY=" ~/.bashrc 2>/dev/null | cut -d= -f2-)
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_CACHE="/home1/doyoonkim/.cache/huggingface/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== eval_full: $RUN_NAME ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL_PATH=$MODEL_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

EXTRA_ARGS=()
[ -n "$BENCHMARKS" ] && EXTRA_ARGS+=(--benchmarks "$BENCHMARKS")
[ "$SKIP_PPL" = "true" ] && EXTRA_ARGS+=(--skip_ppl)
[ "$SKIP_ZEROSHOT" = "true" ] && EXTRA_ARGS+=(--skip_zeroshot)

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --method "$METHOD" \
    --sparsity "$SPARSITY" \
    --gpu_util 0.85 \
    --tp_size "$TP_SIZE" \
    --seed "$SEED" \
    --out_base "$LOCAL_JOB_BASE/eval_${RUN_NAME}" \
    "${EXTRA_ARGS[@]}"

echo "##### END #####"
