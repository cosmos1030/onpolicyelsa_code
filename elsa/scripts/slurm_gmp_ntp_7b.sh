#!/bin/bash
# GMP NTP sweep for DeepSeek-R1-Distill-Qwen-7B (s50 + s60 + s70)
# 4x A100-80GB per job, FSDP sharding
# global_batch = 4 GPUs × batch_size=1 × grad_accum=2 = 8
# 3 LR values × 3 sparsities = 9 jobs total
# Phase 1: torchrun (FSDP train + save), Phase 2: SLURM-level lighteval TP=4

set -e
cd "$(dirname "$0")/.."
SUBMIT_DIR=$(pwd)

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"
GMP_SAVE_BASE="/home1/doyoonkim/projects/elsa/models"

LR_VALUES=(5e-5 1e-4 2e-4)
SPARSITIES=(0.5 0.6 0.7)

mkdir -p tmp_scripts

for SP in "${SPARSITIES[@]}"; do
    SP_TAG=$(echo $SP | tr -d '.')
    for LR in "${LR_VALUES[@]}"; do
        LR_TAG=$(echo $LR | sed 's/e-/em/')
        JOB_NAME="gmp7b_s${SP_TAG}_lr${LR_TAG}"
        SCRIPT="tmp_scripts/${JOB_NAME}.sh"

        cat > $SCRIPT << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --exclude=n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/${JOB_NAME}_%j.out
exec 2>&1

mkdir -p /local-data/user-data/\$USER/job_\$SLURM_JOB_ID/slurm
mkdir -p /local-data/user-data/\$USER/job_\$SLURM_JOB_ID/wandb

export WANDB_DIR=/home1/doyoonkim/projects/elsa/wandb_offline
export WANDB_API_KEY=\$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export WANDB_MODE=offline
mkdir -p \${WANDB_DIR}
export HF_TOKEN=\$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MASTER_PORT=\$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
GMP_SAVE_PATH=${GMP_SAVE_BASE}

echo "=== Job ${JOB_NAME} | sp=${SP} lr=${LR} ==="
echo "SLURM_JOB_ID=\$SLURM_JOB_ID"
echo "NODE: \$(hostname)"
echo "MASTER_PORT=\$MASTER_PORT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd ${SUBMIT_DIR}

# Phase 1: FSDP training + save (eval_math500=false to avoid vLLM/torchrun TCP conflict)
/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun \\
    --nproc_per_node=4 \\
    --master_port=\${MASTER_PORT} \\
    main.py \\
    --model=${MODEL} \\
    --dataset=mixed_cot \\
    --data_path=${DATA_PATH} \\
    --sparsity_ratio=${SP} \\
    --do_gmp=true \\
    --gmp_use_fsdp=true \\
    --gmp_steps=1024 \\
    --gmp_batch_size=1 \\
    --gmp_grad_accum=2 \\
    --gmp_lr=${LR} \\
    --gmp_warmup_ratio=0.05 \\
    --gmp_mask_interval=32 \\
    --gmp_fisher_beta=0.999 \\
    --gmp_kd_lambda=0.0 \\
    --gmp_save_path=\${GMP_SAVE_PATH} \\
    --gmp_max_prompt_len=512 \\
    --gmp_max_seq_len=2048 \\
    --save_model=true \\
    --eval_math500=false \\
    --eval_zero_shot=false \\
    --wandb=true \\
    --wandb_project=gmp_qwen3_7b \\
    --push_to_hub=false \\
    --seed=42

TORCHRUN_EXIT=\$?
echo "=== torchrun exit code: \$TORCHRUN_EXIT ==="
if [ \$TORCHRUN_EXIT -ne 0 ]; then
    echo "ERROR: training failed"
    exit \$TORCHRUN_EXIT
fi

# Phase 1.5: sync offline wandb run to cloud immediately after training
OFFLINE_RUN=\$(ls -td \${WANDB_DIR}/wandb/offline-run-* 2>/dev/null | head -1)
if [ -n "\$OFFLINE_RUN" ]; then
    echo "=== Syncing wandb offline run: \$OFFLINE_RUN ==="
    /home1/doyoonkim/miniconda3/envs/rac/bin/wandb sync \$OFFLINE_RUN || echo "WARNING: wandb sync failed"
fi
unset WANDB_MODE

# Phase 2: lighteval directly from SLURM bash (clean process, TP=4 works)
SAVED_MODEL=\$(ls -td \${GMP_SAVE_PATH}/gmp_* 2>/dev/null | head -1)
if [ -z "\$SAVED_MODEL" ]; then
    echo "ERROR: no saved model found in \${GMP_SAVE_PATH}"
    exit 1
fi
echo "=== Running math500 eval on \$SAVED_MODEL ==="

export CUDA_VISIBLE_DEVICES=0,1,2,3
FREE_MEM=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
TOTAL_MEM=\$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_UTIL=\$(python -c "print(f'{int(\${FREE_MEM}) / int(\${TOTAL_MEM}) * 0.95:.4f}')")
echo "GPU_UTIL=\$GPU_UTIL"

/home1/doyoonkim/miniconda3/envs/rac/bin/lighteval vllm \\
    "model_name=\${SAVED_MODEL},dtype=bfloat16,trust_remote_code=true,tensor_parallel_size=4,gpu_memory_utilization=\${GPU_UTIL},max_model_length=32768,max_num_batched_tokens=32768,seed=42,override_chat_template=true,generation_parameters={max_new_tokens:8192,temperature:0.6,top_p:0.95}" \\
    "lighteval|math_500|0|0" \\
    --output-dir "\${SAVED_MODEL}/lighteval_math500"

LIGHTEVAL_EXIT=\$?
echo "=== lighteval exit code: \$LIGHTEVAL_EXIT ==="

# Phase 3: log math500 to the same wandb run (now online after sync)
if [ \$LIGHTEVAL_EXIT -eq 0 ] && [ -f "\${SAVED_MODEL}/.eval_ctx.json" ]; then
    export SAVED_MODEL_PATH=\$SAVED_MODEL
    /home1/doyoonkim/miniconda3/envs/rac/bin/python << 'PYEOF'
import json, glob, wandb, os, sys
saved_model = os.environ["SAVED_MODEL_PATH"]
with open(f"{saved_model}/.eval_ctx.json") as f:
    ctx = json.load(f)
run_id = ctx.get("wandb_run_id")
project = ctx.get("wandb_project", "gmp_qwen3_7b")
if not run_id:
    print("No run_id found, skipping wandb logging")
    sys.exit(0)
results_files = sorted(glob.glob(f"{saved_model}/lighteval_math500/results/**/*.json", recursive=True))
if not results_files:
    print("No lighteval results files found")
    sys.exit(0)
with open(results_files[-1]) as f:
    results = json.load(f)
task_results = results.get("results", {})
task_key = next((k for k in task_results if "math_500" in k), None)
metric_key = next((k for k in task_results[task_key] if "pass" in k.lower()), None) if task_key else None
if not metric_key:
    print(f"Could not find pass metric. Keys: {list(task_results.keys())}")
    sys.exit(0)
pass_at_1 = task_results[task_key][metric_key]
print(f"math500 pass@1 = {pass_at_1}")
run = wandb.init(project=project, id=run_id, resume="must")
run.log({"math500_pass@1": pass_at_1})
run.finish()
print("wandb logging done")
PYEOF
fi

echo "=== DONE: ${JOB_NAME} ==="
EOF

        chmod +x $SCRIPT
        JID=$(sbatch --parsable $SCRIPT)
        echo "Submitted: ${JOB_NAME} → job $JID (sp=${SP}, lr=${LR})"
    done
done

echo ""
echo "All 9 jobs submitted. Monitor with: squeue -u $USER"
