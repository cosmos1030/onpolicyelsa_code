#!/bin/bash
#SBATCH --job-name=eval_bench_only
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_bench_only_%j.out
exec 2>&1

# Lighteval-only eval (no PPL, no zero-shot harness) for an existing dense or
# saved checkpoint -- skips straight to run_lighteval_bench(), which is what
# you want for re-checking a benchmark after a config change (e.g. bumping
# max_new_tokens, top_k) or for multi-seed variance runs (AIME24/25, GPQA all
# need 3 seeds -- see DATASETS.md / this session's notes: math500/ifeval/
# gsm8k/lcb are fine at 1 seed given their sample counts, AIME/GPQA are not).
#
# Usage: sbatch [--partition=... --gres=gpu:N] slurm_eval_bench_only.sh \
#          <MODEL_PATH> <WANDB_PROJECT> <RUN_NAME> <BENCHMARKS> [SEED] [TP_SIZE]
# e.g. (AIME24/25 at seed 0, tensor_parallel=2, on A5000):
#   sbatch --partition=A5000 --qos=normal --gres=gpu:2 --cpus-per-task=16 --mem=96G \
#     slurm_eval_bench_only.sh \
#     /path/to/Qwen3-8B reasoning_qwen3_8b aime_seed0_8b aime24,aime25 0 2
#
# Current lighteval settings (lib/lighteval_bench.py, as of 2026-08-10):
#   math500/gpqa/ifeval/lcb: max_new_tokens=max_model_length=32768
#   aime24/aime25:           max_new_tokens=max_model_length=38912
#   gsm8k:                   max_new_tokens=2048, max_model_length=4096 (not
#                            in Qwen3's official post-training thinking-mode
#                            suite at all, so no official budget to match)
#   sampling: temperature=0.6, top_p=0.95, top_k=20 (Qwen3 thinking-mode recipe)

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_bench_only.sh <MODEL_PATH> <WANDB_PROJECT> <RUN_NAME> <BENCHMARKS> [SEED] [TP_SIZE]"}
WANDB_PROJECT=${2:?"WANDB_PROJECT required"}
RUN_NAME=${3:?"RUN_NAME required"}
BENCHMARKS=${4:?"BENCHMARKS required, e.g. aime24,aime25 or math500,gpqa,ifeval,lcb"}
SEED=${5:-42}
TP_SIZE=${6:-1}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "=== eval_bench_only: $RUN_NAME (benchmarks=$BENCHMARKS, seed=$SEED, tp_size=$TP_SIZE) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL_PATH=$MODEL_PATH"

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --method dense \
    --sparsity 0.0 \
    --gpu_util 0.85 \
    --tp_size ${TP_SIZE} \
    --skip_ppl \
    --skip_zeroshot \
    --benchmarks "$BENCHMARKS" \
    --seed ${SEED} \
    --out_base /local-data/user-data/${USER}/eval_bench_only_${SLURM_JOB_ID}

echo "##### END #####"
