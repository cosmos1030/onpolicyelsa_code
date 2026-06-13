#!/bin/bash
# Wrapper called by wandb sweep: runs pruning via grpo.py, then evaluates the pruned model.
# Eval metrics are logged back to the same wandb sweep run.
# All args are forwarded to grpo.py as-is.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

cd "${REPO_ROOT}"

########################################
# 1. Capture the sweep run ID (set by wandb agent)
#    so we can log eval metrics back to it later.
########################################
SWEEP_RUN_ID="${WANDB_RUN_ID:-}"
SWEEP_PROJECT="${WANDB_PROJECT:-rac_qwen3_0.6_pruning}"
SWEEP_ENTITY="${WANDB_ENTITY:-dyk6208-gwangju-institute-of-science-and-technology}"

########################################
# 2. Run pruning, capture save path
########################################
PRUNE_LOG=$(mktemp /tmp/prune_log.XXXXXX)

/home1/doyoonkim/miniconda3/envs/rac/bin/python src/open_r1/grpo.py "$@" \
  2>&1 | tee "${PRUNE_LOG}"

MODEL_DIR=$(grep "Pruned model written to" "${PRUNE_LOG}" | tail -1 | sed 's/.*Pruned model written to //')
rm -f "${PRUNE_LOG}"

if [ -z "${MODEL_DIR}" ]; then
  echo "ERROR: Could not find pruned model path in pruning output — skipping eval."
  exit 1
fi

echo "=== Pruned model: ${MODEL_DIR} ==="

########################################
# 3. Evaluate with lighteval + vllm
#    Use a temp wandb project so lighteval doesn't
#    pollute the sweep run; we'll log results ourselves.
########################################
MODEL_TAG=$(basename "${MODEL_DIR}")
OUTPUT_DIR="${REPO_ROOT}/eval_results/${MODEL_TAG}"

MODEL_ARGS="model_name=${MODEL_DIR},\
dtype=bfloat16,\
trust_remote_code=true,\
tensor_parallel_size=1,\
pipeline_parallel_size=1,\
gpu_memory_utilization=0.9,\
max_model_length=32768,\
override_chat_template=true,\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

# Isolate lighteval's own wandb init from the sweep env
unset WANDB_RUN_ID
unset WANDB_SWEEP_ID

echo "=== Starting evaluation ==="
lighteval vllm "${MODEL_ARGS}" "lighteval|math_500|0|0" \
  --output-dir "${OUTPUT_DIR}" \
  --save-details

echo "=== Evaluation finished. Results in ${OUTPUT_DIR} ==="

########################################
# 4. Parse results JSON and log back to the sweep run
########################################
RESULTS_JSON=$(find "${OUTPUT_DIR}" -name "results_*.json" | sort | tail -1)

if [ -z "${RESULTS_JSON}" ]; then
  echo "WARNING: No results JSON found — skipping wandb logging."
  exit 0
fi

echo "=== Logging eval metrics to sweep run ${SWEEP_RUN_ID} ==="

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<PYEOF
import json, wandb, os

with open("${RESULTS_JSON}") as f:
    data = json.load(f)

results = data.get("results", {})

# Flatten: "lighteval|math_500|0" -> "eval/math_500/pass@1"
flat = {}
for task, metrics in results.items():
    if task == "all":
        prefix = "eval/all"
    else:
        # e.g. "lighteval|math_500|0" -> "eval/math_500"
        parts = task.split("|")
        prefix = "eval/" + parts[1] if len(parts) >= 2 else "eval/" + task
    for metric, value in metrics.items():
        # "pass@k:k=1&n=1" -> "pass_at_1"
        key = metric.replace("pass@k:k=1&n=1", "pass_at_1").replace(":", "_").replace("&", "_")
        flat[f"{prefix}/{key}"] = value

run_id = "${SWEEP_RUN_ID}"
project = "${SWEEP_PROJECT}"
entity = "${SWEEP_ENTITY}"

if run_id:
    run = wandb.init(
        id=run_id,
        project=project,
        entity=entity,
        resume="allow",
    )
else:
    run = wandb.init(
        project=project,
        entity=entity,
        name="${MODEL_TAG}",
    )

run.log(flat)
run.finish()
print("Logged to wandb:", flat)
PYEOF