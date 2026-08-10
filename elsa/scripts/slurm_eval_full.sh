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
# PROFILE (lib/lighteval_bench.py) picks the budget/task-set:
#   full (default here):  math500/gpqa/ifeval/lcb @ 32768, aime24/aime25 @
#                          38912 (AIME24/25 only exist in this profile),
#                          gsm8k @ 2048/4096. ~2-4x slower per benchmark than
#                          quick even with tensor_parallel_size=2 (measured:
#                          math500 1002.8s->~2316s, gpqa 617.6s->~1237s for
#                          Qwen3-1.7B) -- use for a checkpoint that's actually
#                          going in a results table.
#   quick:                 math500/gpqa/ifeval/lcb @ 8192, gsm8k @ 2048/4096,
#                          no AIME. What every sweep job used before
#                          2026-08-10 and still the right choice for ranking
#                          configs within one model size cheaply -- NOT
#                          reliable for cross-model-size comparisons (that's
#                          what motivated adding "full"; see DATASETS.md).
#   sampling (both profiles): temperature=0.6, top_p=0.95, top_k=20 (Qwen3
#   thinking-mode recipe -- top_k costs nothing extra, so both profiles use it).
#
#   Seeds: math500/ifeval/gsm8k/lcb are fine at 1 seed given their sample
#   counts; AIME24/25 (30 problems each) and GPQA (198 problems) need 3 seeds
#   (e.g. 42/0/1) -- 1 problem is a multi-point swing on AIME specifically.
#   Pass multiple comma-separated seeds (e.g. "42,0,1") in the SEED slot to
#   run all of them back to back in one job and auto-log per-seed values plus
#   "<metric>_mean"/"<metric>_std" to the same wandb run (see eval_full.py
#   --seeds) -- no manual wandb-API aggregation needed afterward.
#
#   For the full two-stage "sweep cheap, re-verify the winner properly"
#   workflow (quick sweep -> full re-eval of just the best config) see
#   scripts/slurm_eval_final_protocol.sh instead of assembling it by hand here.
#
# Usage: sbatch slurm_eval_full.sh <MODEL_PATH> <RUN_NAME> <METHOD> <SPARSITY> <WANDB_PROJECT> [BENCHMARKS] [SEED_OR_SEEDS] [TP_SIZE] [SKIP_PPL] [SKIP_ZEROSHOT] [PROFILE]
# e.g. (full eval, unchanged from before):
#   sbatch slurm_eval_full.sh /path/to/model sgpt_s60 sparsegpt 0.6 reasoning_qwen3_4b
# e.g. (AIME24/25, 3-seed variance run, tensor_parallel=2 -- pass GPU count via --gres too):
#   sbatch --gres=gpu:2 --cpus-per-task=16 --mem=96G slurm_eval_full.sh \
#     /path/to/Qwen3-8B aime_3seed_8b dense 0.0 reasoning_qwen3_8b aime24,aime25 42,0,1 2 true true full
# e.g. (quick sweep-ranking eval, no AIME, 8192 budget):
#   sbatch slurm_eval_full.sh /path/to/checkpoint sweep_lr5e-5_mi32 gmp 0.6 \
#     reasoning_qwen3_1.7b math500,gpqa,ifeval,lcb,gsm8k 42 1 false false quick

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_full.sh <MODEL_PATH> <RUN_NAME> <METHOD> <SPARSITY> <WANDB_PROJECT> [BENCHMARKS] [SEED_OR_SEEDS] [TP_SIZE] [SKIP_PPL] [SKIP_ZEROSHOT] [PROFILE]"}
RUN_NAME=${2:-"eval"}
METHOD=${3:-"sparsegpt"}
SPARSITY=${4:-0.0}
WANDB_PROJECT=${5:-"reasoning_pruning_v2"}
BENCHMARKS=${6:-}
SEED_OR_SEEDS=${7:-42}
TP_SIZE=${8:-1}
SKIP_PPL=${9:-false}
SKIP_ZEROSHOT=${10:-false}
PROFILE=${11:-full}

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
EXTRA_ARGS+=(--profile "$PROFILE")
if [[ "$SEED_OR_SEEDS" == *,* ]]; then
    EXTRA_ARGS+=(--seeds "$SEED_OR_SEEDS")
else
    EXTRA_ARGS+=(--seed "$SEED_OR_SEEDS")
fi

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --method "$METHOD" \
    --sparsity "$SPARSITY" \
    --gpu_util 0.85 \
    --tp_size "$TP_SIZE" \
    --out_base "$LOCAL_JOB_BASE/eval_${RUN_NAME}" \
    "${EXTRA_ARGS[@]}"

echo "##### END #####"
