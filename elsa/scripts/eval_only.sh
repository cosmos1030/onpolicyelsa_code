#!/bin/bash
# Eval-only script called by wandb sweep agent.
# Runs lighteval+vllm on a pre-saved pruned model and logs results back to the sweep run.
# Args passed by sweep: --model_dir=<path>

set -euo pipefail

REPO_ROOT="/home1/doyoonkim/projects/elsa"
cd "${REPO_ROOT}"

# Parse --model_dir=... from sweep args
MODEL_DIR=""
for arg in "$@"; do
    case "${arg}" in
        --model_dir=*) MODEL_DIR="${arg#--model_dir=}" ;;
    esac
done

if [ -z "${MODEL_DIR}" ]; then
    echo "ERROR: --model_dir not provided"
    exit 1
fi

if [ ! -d "${MODEL_DIR}" ]; then
    echo "ERROR: Model directory not found: ${MODEL_DIR}"
    exit 1
fi

SWEEP_RUN_ID="${WANDB_RUN_ID:-}"
MODEL_TAG=$(basename "${MODEL_DIR}")
OUTPUT_DIR="${REPO_ROOT}/eval_results/${MODEL_TAG}"

echo "=== Evaluating: ${MODEL_TAG} ==="

MODEL_ARGS="model_name=${MODEL_DIR},\
dtype=bfloat16,\
trust_remote_code=true,\
tensor_parallel_size=1,\
pipeline_parallel_size=1,\
gpu_memory_utilization=0.9,\
max_model_length=32768,\
override_chat_template=true,\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

unset WANDB_RUN_ID
unset WANDB_SWEEP_ID

/home1/doyoonkim/miniconda3/envs/rac/bin/lighteval vllm "${MODEL_ARGS}" "lighteval|math_500|0|0" \
    --output-dir "${OUTPUT_DIR}" \
    --save-details

echo "=== Evaluation finished. Results in ${OUTPUT_DIR} ==="

RESULTS_JSON=$(find "${OUTPUT_DIR}" -name "results_*.json" | sort | tail -1)

if [ -z "${RESULTS_JSON}" ]; then
    echo "WARNING: No results JSON found — skipping wandb logging."
    exit 0
fi

echo "=== Logging eval metrics to wandb run ${SWEEP_RUN_ID} ==="

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<PYEOF
import json, wandb

with open("${RESULTS_JSON}") as f:
    data = json.load(f)

flat = {}
for task, metrics in data.get("results", {}).items():
    if task == "all":
        prefix = "eval/all"
    else:
        parts = task.split("|")
        prefix = "eval/" + parts[1] if len(parts) >= 2 else "eval/" + task
    for metric, value in metrics.items():
        key = metric.replace("pass@k:k=1&n=1", "pass_at_1").replace(":", "_").replace("&", "_")
        flat[f"{prefix}/{key}"] = value

run_id = "${SWEEP_RUN_ID}"
if run_id:
    run = wandb.init(
        id=run_id,
        project="elsa_qwen3_0.6",
        entity="dyk6208-gwangju-institute-of-science-and-technology",
        resume="allow",
    )
else:
    run = wandb.init(
        project="elsa_qwen3_0.6",
        entity="dyk6208-gwangju-institute-of-science-and-technology",
        name="${MODEL_TAG}",
    )

run.log(flat)
run.finish()
print("Logged to wandb:", flat)
PYEOF
