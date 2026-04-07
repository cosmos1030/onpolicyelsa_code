#!/bin/bash
#SBATCH --job-name=qwen3_eval_ckpts
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/%j_eval_ckpts.out
#SBATCH -t 2-00:00:00

# Usage: sbatch scripts/slurm_qwen3_dense_eval_ckpts.sh <MODEL_DIR>
# MODEL_DIR: GKD 학습 output 디렉토리 (checkpoint-* 폴더들이 있는 곳)
# Example: sbatch scripts/slurm_qwen3_dense_eval_ckpts.sh models/Qwen3-0.6B_dense_gkd_220k_lr1e-5_steps30

set -euo pipefail
mkdir -p logs

########################################
# 1. Conda
########################################
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

########################################
# 2. Target directory
########################################
MODEL_DIR=${1:-""}
if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_DIR not provided"
    echo "Usage: sbatch slurm_qwen3_dense_eval_ckpts.sh <MODEL_DIR>"
    exit 1
fi

# 절대경로 변환
if [[ "$MODEL_DIR" != /* ]]; then
    MODEL_DIR="/home1/doyoonkim/projects/RAC/open-r1-main/${MODEL_DIR}"
fi

########################################
# 3. Env & wandb run ID 로드
########################################
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

cd /home1/doyoonkim/projects/RAC/open-r1-main

WANDB_RUN_ID_FILE="${MODEL_DIR}/wandb_run_id.txt"
WANDB_RUN_ID=""
if [ -f "$WANDB_RUN_ID_FILE" ]; then
    WANDB_RUN_ID=$(cat "$WANDB_RUN_ID_FILE" | tr -d '[:space:]')
    echo "=== Resuming wandb run: ${WANDB_RUN_ID} ==="
else
    echo "=== WARNING: wandb_run_id.txt not found, eval will log to a new run ==="
fi

########################################
# 4. checkpoint-* + 최종 모델 목록 수집
########################################
EVAL_TARGETS=()

# checkpoint-N 폴더들 (step 순 정렬)
for ckpt in $(ls -d ${MODEL_DIR}/checkpoint-* 2>/dev/null | sort -V); do
    EVAL_TARGETS+=("$ckpt")
done

# 최종 모델 (checkpoint 없이 저장된 경우)
if [ -f "${MODEL_DIR}/config.json" ]; then
    EVAL_TARGETS+=("${MODEL_DIR}")
fi

if [ ${#EVAL_TARGETS[@]} -eq 0 ]; then
    echo "ERROR: No checkpoints found in ${MODEL_DIR}"
    exit 1
fi

echo "=== Found ${#EVAL_TARGETS[@]} checkpoints to evaluate ==="
for t in "${EVAL_TARGETS[@]}"; do echo "  $t"; done

########################################
# 5. 각 체크포인트 평가
########################################
RESULTS_SUMMARY="${MODEL_DIR}/eval_summary.txt"
echo "step,math500_pass@1" > "$RESULTS_SUMMARY"
export WANDB_RUN_ID

for MODEL_PATH in "${EVAL_TARGETS[@]}"; do
    MODEL_TAG=$(basename "$MODEL_PATH")
    OUTPUT_DIR="/home1/doyoonkim/projects/RAC/open-r1-main/eval_results/$(basename ${MODEL_DIR})/${MODEL_TAG}"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "=== Evaluating: ${MODEL_TAG} ==="

    MODEL_ARGS="model_name=${MODEL_PATH},\
dtype=bfloat16,\
trust_remote_code=true,\
tensor_parallel_size=1,\
gpu_memory_utilization=0.9,\
max_model_length=32768,\
override_chat_template=true,\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

    export WANDB_PROJECT=qwen3_dense_gkd_eval
    export WANDB_NAME=$(basename ${MODEL_DIR})_${MODEL_TAG}

    lighteval vllm "$MODEL_ARGS" "lighteval|math_500|0|0" \
        --wandb \
        --output-dir "$OUTPUT_DIR" \
        --save-details

    # 결과 파싱 후 summary + wandb 기록
    RESULT_JSON=$(find "$OUTPUT_DIR" -name "results_*.json" | sort | tail -1)
    if [ -n "$RESULT_JSON" ]; then
        STEP=$(echo "$MODEL_TAG" | grep -oP '\d+' || echo "0")
        python3 - <<PYEOF
import json, wandb, os

with open("${RESULT_JSON}") as f:
    d = json.load(f)
score = d.get("results", {}).get("lighteval|math_500|0", {}).get("pass@k:k=1&n=1", None)
step = int("${STEP}") if "${STEP}".isdigit() else 0

# eval_summary.txt에 기록
with open("${RESULTS_SUMMARY}", "a") as f:
    f.write(f"{step},{score}\n")
print(f"  => step={step}, MATH-500 pass@1: {score}")

# 학습 run에 resume해서 기록 (define_metric으로 eval 전용 step 축 사용)
run_id = "${WANDB_RUN_ID}"
if run_id and score is not None:
    run = wandb.init(
        project="qwen3_gkd",
        id=run_id,
        resume="allow",
    )
    run.define_metric("eval/step")
    run.define_metric("eval/*", step_metric="eval/step")
    run.log({"eval/step": step, "eval/math500_pass@1": score})
    run.finish()
PYEOF
    fi
done

echo ""
echo "=== All evaluations done ==="
echo "Summary saved to: ${RESULTS_SUMMARY}"
cat "$RESULTS_SUMMARY"

date
echo "##### END #####"
