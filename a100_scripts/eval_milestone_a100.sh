#!/bin/bash
# Score ONE milestone checkpoint on an A100 80GB. Run this on the A100 box.
#
# Usage:  bash a100_scripts/eval_milestone_a100.sh <HF_REPO> [WANDB_RUN_ID]
#   e.g.  bash a100_scripts/eval_milestone_a100.sh cosmos1030/alps4b-s70-2term-step000512
#
# Env:
#   TP=1            tensor_parallel_size. 1 fits Qwen3-4B bf16 on one 80GB A100
#                   with room for the 8192-token KV budget. Set TP=4 to mirror
#                   the tp=4 path the 2048 endpoints were scored on.
#   GPU_UTIL=0.90   vLLM gpu_memory_utilization.
#   WORK=...        where to download weights (default ./a100_eval_work)
#   CUDA_VISIBLE_DEVICES=<n>   which GPU(s) to use.
#
# Budgets come from lighteval_bench.py's "quick" profile and are hardware
# independent: math500/gpqa/ifeval/lcb at max_new_tokens=8192, gsm8k at 2048.
# Scores land in $WORK/<name>/eval_summary_resumed.json and in the stdout tail.
set -euo pipefail

REPO=${1:?"Usage: $0 <HF_REPO> [WANDB_RUN_ID]"}
WANDB_RUN_ID=${2:-}
NAME=$(basename "$REPO")
WORK=${WORK:-$PWD/a100_eval_work}
TP=${TP:-1}
GPU_UTIL=${GPU_UTIL:-0.90}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

export HF_TOKEN=${HF_TOKEN:?"export HF_TOKEN=<your token> first"}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0            # matches the path every other number was scored on
export VLLM_HOST_IP=127.0.0.1
export VLLM_NO_USAGE_STATS=1
export NCCL_DEBUG=WARN
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE   # lighteval pulls tasks from the hub

MODEL_DIR="$WORK/$NAME"
mkdir -p "$WORK"
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "=== downloading $REPO -> $MODEL_DIR ==="
  hf download "$REPO" --local-dir "$MODEL_DIR"
fi

echo "=== eval $NAME  (tp=$TP gpu_util=$GPU_UTIL gpus=${CUDA_VISIBLE_DEVICES:-all}) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --no_hub: the weights are already on the Hub, re-uploading them is pointless.
# WANDB_RUN_ID is optional and OFF by default on purpose -- three milestones
# resumed into one parent run overwrite each other's run.summary (avg5,
# math500_pass@1), so the trajectory cannot be read back from wandb. Take the
# numbers from eval_summary_resumed.json instead.
WB=()
[ -n "$WANDB_RUN_ID" ] && WB=(--wandb_run_id "$WANDB_RUN_ID")

cd "$REPO_ROOT/elsa"
exec python "$REPO_ROOT/b200_scripts/resume_eval_lighteval.py" \
    --model_dir "$MODEL_DIR" \
    --tp_size "$TP" --gpu_util "$GPU_UTIL" \
    --profile quick --seed 42 --no_hub "${WB[@]}"
