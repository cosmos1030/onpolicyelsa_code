#!/bin/bash
#SBATCH --job-name=eval_5bench
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/eval_5bench_%j.out
exec 2>&1

# Usage:
#   sbatch slurm_eval_5bench.sh <MODEL_PATH> [WANDB_RUN_ID] [WANDB_PROJECT]
#
# Runs MATH-500, GPQA-Diamond, MMLU-Redux, IFEval, LiveCodeBench sequentially,
# then logs all results to wandb (resuming existing run if WANDB_RUN_ID is given).
#
# Temperature=0.6, top_p=0.95 (matches ReasoningQAT paper protocol)
# max_new_tokens=32768 for reasoning tasks, 32768 for LCB
# MMLU-Redux and IFEval use max_new_tokens=4096 (short-answer tasks)

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_5bench.sh <MODEL_PATH> [WANDB_RUN_ID] [WANDB_PROJECT]"}
WANDB_RUN_ID=${2:-""}
WANDB_PROJECT=${3:-"elsa_eval"}

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb

export WANDB_DIR=/local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1 | tr -d ' \n\r')
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

LIGHTEVAL=/home1/doyoonkim/miniconda3/envs/rac/bin/lighteval
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

echo "=== eval_5bench ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "WANDB_RUN_ID=$WANDB_RUN_ID"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Compute GPU memory utilization dynamically
GPU_UTIL=$($PYTHON -c "
import torch
free, total = torch.cuda.mem_get_info(0)
print(f'{free/total*0.92:.4f}')
")
echo "GPU_UTIL=$GPU_UTIL"

OUTPUT_BASE="${MODEL_PATH}/eval_5bench"
mkdir -p "$OUTPUT_BASE"

# MMLU-Redux subset list (57 subsets)
MMLU_SUBSETS=(
    abstract_algebra anatomy astronomy business_ethics clinical_knowledge
    college_biology college_chemistry college_computer_science college_mathematics
    college_medicine college_physics computer_security conceptual_physics
    econometrics electrical_engineering elementary_mathematics formal_logic
    global_facts high_school_biology high_school_chemistry high_school_computer_science
    high_school_european_history high_school_geography high_school_government_and_politics
    high_school_macroeconomics high_school_mathematics high_school_microeconomics
    high_school_physics high_school_psychology high_school_statistics
    high_school_us_history high_school_world_history human_aging human_sexuality
    international_law jurisprudence logical_fallacies machine_learning management
    marketing medical_genetics miscellaneous moral_disputes moral_scenarios
    nutrition philosophy prehistory professional_accounting professional_law
    professional_medicine professional_psychology public_relations security_studies
    sociology us_foreign_policy virology world_religions
)

# Build comma-separated mmlu_redux task string
MMLU_TASKS=""
for s in "${MMLU_SUBSETS[@]}"; do
    MMLU_TASKS="${MMLU_TASKS}lighteval|mmlu_redux_2:${s}|0|0,"
done
MMLU_TASKS="${MMLU_TASKS%,}"  # strip trailing comma

# Common model args
MODEL_ARGS_BASE="model_name=${MODEL_PATH},dtype=bfloat16,trust_remote_code=true,tensor_parallel_size=1,gpu_memory_utilization=${GPU_UTIL},max_model_length=32768,max_num_batched_tokens=32768,seed=42,override_chat_template=true"

run_lighteval() {
    local task_id=$1
    local task_str=$2
    local max_new_tokens=$3
    local out_dir="${OUTPUT_BASE}/${task_id}"
    mkdir -p "$out_dir"
    echo "--- Running: $task_id ---"
    $LIGHTEVAL vllm \
        "${MODEL_ARGS_BASE},generation_parameters={max_new_tokens:${max_new_tokens},temperature:0.6,top_p:0.95}" \
        "$task_str" \
        --output-dir "$out_dir" \
        --save-details
    echo "--- Done: $task_id (exit $?) ---"
}

# 1. MATH-500
run_lighteval "math500"   "lighteval|math_500|0|0"       8192

# 2. GPQA-Diamond
run_lighteval "gpqa"      "lighteval|gpqa:diamond|0|0"   8192

# 3. IFEval
run_lighteval "ifeval"    "lighteval|ifeval|0|0"         4096

# 4. MMLU-Redux (all 57 subsets in one call)
run_lighteval "mmlu_redux" "$MMLU_TASKS"                 4096

# 5. LiveCodeBench
run_lighteval "lcb"       "lighteval|lcb:codegeneration|0|0" 8192

echo "=== All evals done. Parsing results and logging to wandb... ==="

$PYTHON - <<PYEOF
import json, glob, math, os, sys
import wandb

model_path    = '$MODEL_PATH'
wandb_run_id  = '$WANDB_RUN_ID'
wandb_project = '$WANDB_PROJECT'
output_base = os.path.join(model_path, "eval_5bench")


def parse_results(out_dir):
    files = sorted(glob.glob(f"{out_dir}/**/results_*.json", recursive=True))
    if not files:
        print(f"  [WARN] no results_*.json in {out_dir}")
        return {}
    with open(files[-1]) as f:
        return json.load(f).get("results", {})


def safe(v):
    return float(v) if (v is not None and not (isinstance(v, float) and math.isnan(v))) else float("nan")


metrics = {}

# --- MATH-500 ---
r = parse_results(os.path.join(output_base, "math500"))
task = r.get("lighteval|math_500|0", {})
metrics["eval/math500_pass@1"]        = safe(task.get("pass@k:k=1&n=1"))
metrics["eval/math500_pass@1_stderr"] = safe(task.get("pass@k:k=1&n=1_stderr"))
print(f"MATH-500:     {metrics['eval/math500_pass@1']:.4f}")

# --- GPQA-Diamond ---
r = parse_results(os.path.join(output_base, "gpqa"))
task = r.get("lighteval|gpqa:diamond|0", {})
# lighteval reports acc_norm or acc depending on version
gpqa_score = task.get("acc_norm", task.get("acc", task.get("pass@k:k=1&n=1")))
metrics["eval/gpqa_diamond_acc"]        = safe(gpqa_score)
metrics["eval/gpqa_diamond_acc_stderr"] = safe(task.get("acc_norm_stderr", task.get("acc_stderr")))
print(f"GPQA-Diamond: {metrics['eval/gpqa_diamond_acc']:.4f}")

# --- IFEval ---
r = parse_results(os.path.join(output_base, "ifeval"))
task = r.get("lighteval|ifeval|0", {})
# IFEval reports prompt-level and instruction-level accuracy
ifeval_prompt = task.get("prompt_level_strict_acc", task.get("acc"))
ifeval_inst   = task.get("inst_level_strict_acc")
metrics["eval/ifeval_prompt_acc"]  = safe(ifeval_prompt)
metrics["eval/ifeval_inst_acc"]    = safe(ifeval_inst)
print(f"IFEval (prompt): {metrics['eval/ifeval_prompt_acc']:.4f}")
print(f"IFEval (inst):   {metrics['eval/ifeval_inst_acc']:.4f}")

# --- MMLU-Redux ---
r = parse_results(os.path.join(output_base, "mmlu_redux"))
mmlu_scores = []
for key, val in r.items():
    if "mmlu_redux_2" in key:
        score = val.get("acc_norm", val.get("acc"))
        if score is not None and not (isinstance(score, float) and math.isnan(score)):
            mmlu_scores.append(float(score))
mmlu_avg = sum(mmlu_scores) / len(mmlu_scores) if mmlu_scores else float("nan")
metrics["eval/mmlu_redux_acc"]     = mmlu_avg
metrics["eval/mmlu_redux_n_tasks"] = len(mmlu_scores)
print(f"MMLU-Redux ({len(mmlu_scores)} tasks): {mmlu_avg:.4f}")

# --- LiveCodeBench ---
r = parse_results(os.path.join(output_base, "lcb"))
task = r.get("lighteval|lcb:codegeneration|0", {})
lcb_score = task.get("codegen_pass@1:16", task.get("pass@1"))
metrics["eval/lcb_pass@1"] = safe(lcb_score)
print(f"LCB:          {metrics['eval/lcb_pass@1']:.4f}")

# Save JSON locally
out_json = os.path.join(output_base, "all_results.json")
with open(out_json, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nResults saved to {out_json}")

# Log to wandb
if wandb_run_id:
    run = wandb.init(project=wandb_project, id=wandb_run_id, resume="must")
else:
    run = wandb.init(project=wandb_project, name=os.path.basename(model_path) + "_eval")
run.log(metrics)
run.finish()
print("wandb logging done")
PYEOF

echo "=== DONE ==="
