#!/bin/bash
#SBATCH --job-name=eval_math500
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --exclude=n80
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_math500_%j.out
exec 2>&1

# Usage: sbatch slurm_eval_math500.sh <MODEL_PATH> <WANDB_RUN_ID> <WANDB_PROJECT>
MODEL_PATH=$1
WANDB_RUN_ID=$2
WANDB_PROJECT=${3:-gmp_dpo_1.5b}

if [ -z "$MODEL_PATH" ] || [ -z "$WANDB_RUN_ID" ]; then
    echo "Usage: sbatch slurm_eval_math500.sh <MODEL_PATH> <WANDB_RUN_ID> [WANDB_PROJECT]"
    exit 1
fi

export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_V1=0

echo "=== eval_math500 ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "WANDB_RUN_ID=$WANDB_RUN_ID"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "NODE=$(hostname)"

OUTPUT_DIR="${MODEL_PATH}/lighteval_math500"
mkdir -p "$OUTPUT_DIR"

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_UTIL=$(/home1/doyoonkim/miniconda3/envs/rac/bin/python -c "print(f'{int(${FREE_MEM}) / int(${TOTAL_MEM}) * 0.95:.4f}')")
echo "GPU_UTIL=$GPU_UTIL"

/home1/doyoonkim/miniconda3/envs/rac/bin/lighteval vllm \
    "model_name=${MODEL_PATH},dtype=bfloat16,trust_remote_code=true,tensor_parallel_size=1,gpu_memory_utilization=${GPU_UTIL},max_model_length=32768,max_num_batched_tokens=32768,seed=42,override_chat_template=true,generation_parameters={max_new_tokens:8192,temperature:0.6,top_p:0.95}" \
    "lighteval|math_500|0|0" \
    --output-dir "$OUTPUT_DIR" \
    --max-samples 500

LIGHTEVAL_EXIT=$?
echo "lighteval exit code: $LIGHTEVAL_EXIT"

if [ $LIGHTEVAL_EXIT -ne 0 ]; then
    echo "ERROR: lighteval failed"
    exit $LIGHTEVAL_EXIT
fi

export MODEL_PATH_ENV=$MODEL_PATH
export WANDB_RUN_ID_ENV=$WANDB_RUN_ID
export WANDB_PROJECT_ENV=$WANDB_PROJECT

/home1/doyoonkim/miniconda3/envs/rac/bin/python << 'PYEOF'
import json, glob, wandb, os, sys

model_path = os.environ["MODEL_PATH_ENV"]
run_id = os.environ["WANDB_RUN_ID_ENV"]
project = os.environ["WANDB_PROJECT_ENV"]
output_dir = f"{model_path}/lighteval_math500"

results_files = sorted(glob.glob(f"{output_dir}/**/results_*.json", recursive=True))
print(f"Results files: {results_files}")
if not results_files:
    print("No results files found")
    sys.exit(1)

with open(results_files[-1]) as f:
    results = json.load(f)

task_results = results.get("results", {}).get("lighteval|math_500|0", {})
pass_at_1 = task_results.get("pass@k:k=1&n=1", float("nan"))
stderr = task_results.get("pass@k:k=1&n=1_stderr", float("nan"))
print(f"math500 pass@1 = {pass_at_1:.4f} ± {stderr:.4f}")

run = wandb.init(project=project, id=run_id, resume="must")
run.log({
    "math500_pass@1": pass_at_1,
    "eval/math500_pass@1": pass_at_1,
    "eval/math500_pass@1_stderr": stderr,
    "eval/math500_pass@1_upper": pass_at_1 + stderr,
    "eval/math500_pass@1_lower": pass_at_1 - stderr,
})
run.finish()
print("wandb logging done")
PYEOF

echo "=== DONE ==="
