#!/bin/bash
# Wrapper called by wandb sweep: runs GKD training via gkd.py, then evaluates
# every saved checkpoint with lighteval+vllm.
# Eval metrics are logged back to the same wandb sweep run with correct step numbers.
# All args are forwarded to gkd.py as-is.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

########################################
# 1. Capture sweep run ID before training
########################################
SWEEP_RUN_ID="${WANDB_RUN_ID:-}"
SWEEP_PROJECT="${WANDB_PROJECT:-rac_qwen3_0.6_pruning}"  # must match sweep yaml project
SWEEP_ENTITY="${WANDB_ENTITY:-dyk6208-gwangju-institute-of-science-and-technology}"

# Parse key hyperparams from args to build a human-readable run name
_lr=""; _opt=""; _sched=""
for arg in "$@"; do
  case "$arg" in
    --learning_rate=*)    _lr="${arg#*=}" ;;
    --sparse_optimizer=*) _opt="${arg#*=}" ;;
    --lr_scheduler_type=*) _sched="${arg#*=}" ;;
  esac
done
RUN_NAME="lr${_lr}_opt${_opt}_sched${_sched}"
export WANDB_NAME="${RUN_NAME}"

# Build a unique output dir from the run name
OUTPUT_DIR="${REPO_ROOT}/models/gkd_qwen3_0.6_${RUN_NAME}_${SWEEP_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_TAG=$(basename "${OUTPUT_DIR}")

########################################
# 2. Run GKD training
########################################
export OPEN_R1_SFT_CACHE_DIR=~/.cache/open-r1/sft_datasets/${RUN_NAME}

echo "=== Starting GKD training → ${OUTPUT_DIR} ==="

accelerate launch --config_file recipes/accelerate_1gpu.yaml \
  src/open_r1/gkd.py "$@" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1

echo "=== GKD training finished ==="

########################################
# 3. Evaluate every checkpoint
########################################
unset WANDB_RUN_ID
unset WANDB_SWEEP_ID

# Collect checkpoint dirs sorted by step number
CHECKPOINTS=$(find "${OUTPUT_DIR}" -maxdepth 1 -name "checkpoint-*" -type d | sort -t'-' -k2 -n)

if [ -z "${CHECKPOINTS}" ]; then
  echo "WARNING: No checkpoints found in ${OUTPUT_DIR} — skipping eval."
  exit 0
fi

for CKPT_DIR in ${CHECKPOINTS}; do
  STEP=$(basename "${CKPT_DIR}" | sed 's/checkpoint-//')
  EVAL_OUTPUT_DIR="${REPO_ROOT}/eval_results/${MODEL_TAG}/step_${STEP}"

  echo "=== Evaluating checkpoint step ${STEP}: ${CKPT_DIR} ==="

  MODEL_ARGS="model_name=${CKPT_DIR},\
dtype=bfloat16,\
trust_remote_code=true,\
tensor_parallel_size=1,\
pipeline_parallel_size=1,\
gpu_memory_utilization=0.9,\
max_model_length=32768,\
override_chat_template=true,\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

  lighteval vllm "${MODEL_ARGS}" "lighteval|math_500|0|0" \
    --output-dir "${EVAL_OUTPUT_DIR}" \
    --save-details || echo "WARNING: lighteval failed for step ${STEP} — skipping."

  RESULTS_JSON=$(find "${EVAL_OUTPUT_DIR}" -name "results_*.json" 2>/dev/null | sort | tail -1)
  if [ -z "${RESULTS_JSON}" ]; then
    echo "WARNING: No results JSON for step ${STEP} — skipping wandb logging."
    continue
  fi

  /home1/doyoonkim/miniconda3/envs/rac/bin/python - <<PYEOF
import json, wandb

with open("${RESULTS_JSON}") as f:
    data = json.load(f)

results = data.get("results", {})
flat = {}
for task, metrics in results.items():
    if task == "all":
        prefix = "eval/all"
    else:
        parts = task.split("|")
        prefix = "eval/" + parts[1] if len(parts) >= 2 else "eval/" + task
    for metric, value in metrics.items():
        key = metric.replace("pass@k:k=1&n=1", "pass_at_1").replace(":", "_").replace("&", "_")
        flat[f"{prefix}/{key}"] = value

run_id = "${SWEEP_RUN_ID}"
project = "${SWEEP_PROJECT}"
entity = "${SWEEP_ENTITY}"

if run_id:
    run = wandb.init(id=run_id, project=project, entity=entity, resume="allow")
else:
    run = wandb.init(project=project, entity=entity, name="${MODEL_TAG}")

run.log(flat, step=int(${STEP}))
run.finish()
print(f"Step ${STEP} logged to wandb:", flat)
PYEOF

done

echo "=== All checkpoint evaluations finished ==="