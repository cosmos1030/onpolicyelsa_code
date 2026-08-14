#!/bin/bash
#SBATCH --job-name=eval_lighteval
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/eval_lighteval_%j.out
exec 2>&1

# Standalone lighteval-only re-run (math500/aime24/aime25/gpqa/lcb/ifeval/gsm8k)
# for a training job that already completed + saved its checkpoint but got
# stuck during eval_full_bench -- specifically on lcb:codegeneration's dataset
# download, which hangs forever in huggingface_hub's "xet" transfer backend
# on this cluster's network path. HF_HUB_DISABLE_XET=1 forces the regular
# HTTP downloader instead. PPL/zero-shot are skipped here since they already
# completed and logged in the original run before it got stuck.
#
# Usage: sbatch slurm_eval_lighteval_only.sh <MODEL_DIR_NAME> <WANDB_PROJECT> <WANDB_RUN_ID> [BENCHMARKS]
# BENCHMARKS: optional comma-separated subset of math500,aime24,aime25,gpqa,ifeval,lcb,gsm8k
#        (default: all 7) -- e.g. to re-run just the one benchmark that
#        crashed a prior eval instead of redoing all of them.
# e.g.: sbatch slurm_eval_lighteval_only.sh gmp_s50pct_lr0.0001_20260808_024943 reasoning_qwen3_1.7b bcdyv5co
#       sbatch slurm_eval_lighteval_only.sh gmp_s70pct_..._20260809_204703 reasoning_qwen3_4b duk3az49 gsm8k

MODEL_DIR_NAME=${1:?"Usage: sbatch slurm_eval_lighteval_only.sh <MODEL_DIR_NAME> <WANDB_PROJECT> <WANDB_RUN_ID> [BENCHMARKS]"}
WANDB_PROJECT=${2:?"missing WANDB_PROJECT"}
WANDB_RUN_ID=${3:?"missing WANDB_RUN_ID"}
BENCHMARKS=${4:-}

REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
MODEL_PATH="${REPO_ROOT}/models/${MODEL_DIR_NAME}"
OUT_BASE="/tmp/${USER}/eval_${MODEL_DIR_NAME}"

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_HOME=/home/shared/huggingface
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_HOST_IP=127.0.0.1
export TMPDIR=/tmp

echo "=== lighteval-only re-run: $MODEL_DIR_NAME (wandb $WANDB_PROJECT/$WANDB_RUN_ID) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL_PATH=$MODEL_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO_ROOT"

python scripts/eval_full.py \
    --model_path="$MODEL_PATH" \
    --wandb_project="$WANDB_PROJECT" \
    --wandb_run_id="$WANDB_RUN_ID" \
    --skip_ppl \
    --skip_zeroshot \
    --out_base="$OUT_BASE" \
    --gpu_util=0.85 \
    --seed=42 \
    ${BENCHMARKS:+--benchmarks="$BENCHMARKS"}

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
