#!/bin/bash
#SBATCH --job-name=eval_smoketest
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --exclude=n80
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_smoketest_%j.out
exec 2>&1

# Smoke test: runs all 5 benchmarks with --max-samples 1 to verify the pipeline works.
# Usage: sbatch slurm_eval_5bench_smoketest.sh [MODEL_PATH]

MODEL_PATH=${1:-"/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"}

mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DISABLED=true
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

LIGHTEVAL=/home1/doyoonkim/miniconda3/envs/rac/bin/lighteval
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

echo "=== eval_5bench_smoketest ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

GPU_UTIL=$($PYTHON -c "
import torch
free, total = torch.cuda.mem_get_info(0)
print(f'{free/total*0.92:.4f}')
")
echo "GPU_UTIL=$GPU_UTIL"

OUTPUT_BASE="/tmp/eval_smoketest_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_BASE"

MODEL_ARGS="model_name=${MODEL_PATH},dtype=bfloat16,trust_remote_code=true,tensor_parallel_size=1,gpu_memory_utilization=${GPU_UTIL},max_model_length=8192,max_num_batched_tokens=8192,seed=42,override_chat_template=true,generation_parameters={max_new_tokens:512,temperature:0.6,top_p:0.95}"

run_one() {
    local task_id=$1
    local task_str=$2
    local max_samples=${3:-1}
    local out_dir="${OUTPUT_BASE}/${task_id}"
    mkdir -p "$out_dir"
    echo ""
    echo "========== $task_id =========="
    $LIGHTEVAL vllm \
        "$MODEL_ARGS" \
        "$task_str" \
        --output-dir "$out_dir" \
        --max-samples "$max_samples" \
        --save-details
    local exit_code=$?
    echo "--- $task_id exit code: $exit_code ---"
    # Print result JSON key-values
    $PYTHON -c "
import json, glob
files = sorted(glob.glob('${out_dir}/**/results_*.json', recursive=True))
if not files:
    print('  [WARN] no results_*.json found')
else:
    d = json.load(open(files[-1]))
    for task, vals in d.get('results', {}).items():
        print(f'  {task}:')
        for k, v in vals.items():
            if not k.endswith('_stderr'):
                print(f'    {k} = {v}')
" 2>/dev/null
    return $exit_code
}

PASS=0
FAIL=0

check() {
    local name=$1; shift
    run_one "$@"  # task_id task_str [max_samples]
    if [ $? -eq 0 ]; then
        echo "[PASS] $name"
        PASS=$((PASS+1))
    else
        echo "[FAIL] $name"
        FAIL=$((FAIL+1))
    fi
}

check "MATH-500"   math500    "lighteval|math_500|0|0"                    1
check "GPQA"       gpqa       "lighteval|gpqa:diamond|0|0"               1
check "IFEval"     ifeval     "lighteval|ifeval|0|0"                      1
check "MMLU-Redux" mmlu_redux "lighteval|mmlu_redux_2:abstract_algebra|0|0" 4
check "LCB"        lcb        "lighteval|lcb:codegeneration|0|0"          1

echo ""
echo "=========================================="
echo "SMOKE TEST RESULTS: PASS=$PASS  FAIL=$FAIL"
echo "=========================================="

rm -rf "$OUTPUT_BASE"

[ $FAIL -eq 0 ] && echo "All benchmarks OK." || echo "Some benchmarks FAILED — check logs above."
