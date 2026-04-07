#!/bin/bash
#SBATCH --job-name=elsa_qwen3_kd_2k
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=output_qwen/%j.out
#SBATCH -t 3-00:00:00

mkdir -p output_qwen

########################################
# Paths & Hyperparameters
########################################
MODEL=Qwen/Qwen3-0.6B
KD_DATA=/home1/doyoonkim/projects/elsa/data/math_220k_prompts.jsonl
SAVE_DIR=/home1/doyoonkim/projects/elsa/models

SPARSITY=0.5
ADMM_STEPS=1024
ADMM_LR=1e-2
ADMM_LMDA=5e-5
ADMM_LMDA_SCHEDULE=cosine
ADMM_INTERVAL=32
BS=1
ACCUM=8

KD_MAX_PROMPT_LEN=512
KD_MAX_NEW_TOKENS=2048
KD_TEMPERATURE=1.0

########################################
# Run
########################################
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home1/doyoonkim/miniconda3/envs/rac/bin/python main.py \
  --model="${MODEL}" \
  --sparsity_ratio=${SPARSITY} \
  --sparsity_type=unstructured \
  --do_kd_admm=True \
  --kd_data_path="${KD_DATA}" \
  --kd_max_prompt_len=${KD_MAX_PROMPT_LEN} \
  --kd_max_new_tokens=${KD_MAX_NEW_TOKENS} \
  --kd_temperature=${KD_TEMPERATURE} \
  --kd_topk=50 \
  --admm_steps=${ADMM_STEPS} \
  --admm_batch_size=${BS} \
  --admm_gradient_accumulation_steps=${ACCUM} \
  --admm_lr=${ADMM_LR} \
  --admm_lmda=${ADMM_LMDA} \
  --admm_final_lmda=${ADMM_LMDA} \
  --admm_lmda_schedule_mode=${ADMM_LMDA_SCHEDULE} \
  --admm_beta2=0.999 \
  --admm_interval=${ADMM_INTERVAL} \
  --admm_base_optimizer=adamw \
  --admm_precision=bf16 \
  --save_model=True \
  --admm_save_path="${SAVE_DIR}" \
  --eval_zero_shot=False \
  --wandb=True \
  --wandb_project=elsa_qwen3_0.6b \
  --seed=42

########################################
# Eval: 저장된 모델 경로 찾아서 lighteval 실행
########################################
MODEL_NAME=$(basename ${MODEL})
KD_DATA_TAG=$(basename ${KD_DATA} .jsonl)

# gkd_admm.py의 run_name 패턴과 동일하게 glob
PRUNED_MODEL=$(ls -dt ${SAVE_DIR}/${MODEL_NAME}_pruned${SPARSITY}_kd_${KD_DATA_TAG}_admm_lr${ADMM_LR}_lmda${ADMM_LMDA}_* 2>/dev/null | head -1)

if [ -z "$PRUNED_MODEL" ]; then
    echo "ERROR: Pruned model not found in ${SAVE_DIR}"
    exit 1
fi

echo "=== Evaluating pruned model: ${PRUNED_MODEL} ==="

MODEL_TAG=$(basename "$PRUNED_MODEL")
EVAL_OUTPUT_DIR=/home1/doyoonkim/projects/elsa/eval_results/${MODEL_TAG}
mkdir -p "$EVAL_OUTPUT_DIR"

# 학습 run ID 조회 (gkd_admm.py의 run_name과 동일)
WANDB_RUN_NAME="${MODEL_TAG}"
WANDB_RUN_ID=$(/home1/doyoonkim/miniconda3/envs/elsa/bin/python3 -c "
import wandb
api = wandb.Api()
runs = api.runs('dyk6208-gwangju-institute-of-science-and-technology/elsa_qwen3_0.6b',
                filters={'displayName': '${WANDB_RUN_NAME}'})
runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
print(runs[0].id if runs else '')
" 2>/dev/null)

MODEL_ARGS="model_name=${PRUNED_MODEL},\
dtype=bfloat16,\
trust_remote_code=true,\
tensor_parallel_size=1,\
gpu_memory_utilization=0.9,\
max_model_length=32768,\
override_chat_template=true,\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

/home1/doyoonkim/miniconda3/envs/rac/bin/lighteval vllm "$MODEL_ARGS" "lighteval|math_500|0|0" \
    --output-dir "$EVAL_OUTPUT_DIR" \
    --save-details

# eval 결과를 학습 run에 기록
RESULT_JSON=$(find "$EVAL_OUTPUT_DIR" -name "results_*.json" | sort | tail -1)
if [ -n "$RESULT_JSON" ] && [ -n "$WANDB_RUN_ID" ]; then
    /home1/doyoonkim/miniconda3/envs/elsa/bin/python3 - <<PYEOF
import json, wandb
with open("${RESULT_JSON}") as f:
    d = json.load(f)
score = d.get("results", {}).get("lighteval|math_500|0", {}).get("pass@k:k=1&n=1", None)
print(f"MATH-500 pass@1: {score}")
if score is not None:
    run = wandb.init(
        project="elsa_qwen3_0.6b",
        entity="dyk6208-gwangju-institute-of-science-and-technology",
        id="${WANDB_RUN_ID}",
        resume="allow",
    )
    run.log({"eval/math500_pass@1": score})
    run.finish()
PYEOF
fi

echo "=== Evaluation done. Results in ${EVAL_OUTPUT_DIR} ==="
echo "##### END #####"