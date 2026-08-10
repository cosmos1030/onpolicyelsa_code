#!/bin/bash
#SBATCH --job-name=eval_final_protocol
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_final_protocol_%j.out
exec 2>&1

# Full "official protocol" re-evaluation for a single winning checkpoint
# (picked from a cheap 8192-budget sweep by math500 ranking -- see this
# session's notes: sweeping at 8192 is ~2-4x faster and preserves ranking
# well enough within a fixed model size, but is NOT reliable for absolute
# numbers or cross-model-size comparisons, so only re-run the actual winners
# through this before they go in a table).
#
# Runs eval_full.py TWICE against the same checkpoint into the SAME wandb
# run, because a single call's --seeds applies to every benchmark in that
# call and we want different seed counts per benchmark:
#   1) PPL + zero-shot + math500/ifeval/lcb, single seed (cheap, large
#      sample counts already average out noise)
#   2) gpqa + aime24 + aime25, 3 seeds (42/0/1) -- small sample counts,
#      high per-problem variance, see DATASETS.md / this session's notes
#
# Usage: sbatch [--gres=gpu:2 --cpus-per-task=16 --mem=96G ...] slurm_eval_final_protocol.sh \
#          <MODEL_PATH> <RUN_NAME> <METHOD> <SPARSITY> <WANDB_PROJECT> [TP_SIZE]
# e.g.:
#   sbatch --partition=A100-80GB --qos=hpgpu --gres=gpu:2 --cpus-per-task=16 --mem=96G --time=8:00:00 \
#     slurm_eval_final_protocol.sh /path/to/winning_checkpoint winner_s60_lr5e-5_mi32 gmp 0.6 reasoning_qwen3_1.7b 2

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_final_protocol.sh <MODEL_PATH> <RUN_NAME> <METHOD> <SPARSITY> <WANDB_PROJECT> [TP_SIZE]"}
RUN_NAME=${2:-"eval"}
METHOD=${3:-"gmp"}
SPARSITY=${4:-0.0}
WANDB_PROJECT=${5:-"reasoning_pruning_v2"}
TP_SIZE=${6:-1}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="/home1/doyoonkim/projects/elsa/logs/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY=$(grep "^export WANDB_API_KEY=" ~/.bashrc 2>/dev/null | cut -d= -f2-)
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_CACHE="/home1/doyoonkim/.cache/huggingface/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== eval_final_protocol: $RUN_NAME ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL_PATH=$MODEL_PATH"

cd /home1/doyoonkim/projects/elsa

echo "--- stage 1: PPL + zero-shot + math500/ifeval/lcb (seed=42) ---"
STAGE1_LOG="$LOCAL_JOB_BASE/stage1.log"
$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --method "$METHOD" \
    --sparsity "$SPARSITY" \
    --gpu_util 0.85 \
    --tp_size "$TP_SIZE" \
    --benchmarks math500,ifeval,lcb \
    --profile full \
    --out_base "$LOCAL_JOB_BASE/eval_${RUN_NAME}" \
    2>&1 | tee "$STAGE1_LOG"

RUN_ID=$(grep -oP '(?<=\[eval_full\] wandb run id: )\S+' "$STAGE1_LOG" | tail -1)
if [ -z "$RUN_ID" ]; then
    echo "ERROR: could not find wandb run id in stage 1 output, aborting stage 2."
    exit 1
fi
echo "stage 1 wandb run id: $RUN_ID"

echo "--- stage 2: gpqa + aime24 + aime25 (seeds=42,0,1), resuming run $RUN_ID ---"
$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_id "$RUN_ID" \
    --method "$METHOD" \
    --sparsity "$SPARSITY" \
    --gpu_util 0.85 \
    --tp_size "$TP_SIZE" \
    --skip_ppl \
    --skip_zeroshot \
    --benchmarks gpqa,aime24,aime25 \
    --seeds 42,0,1 \
    --profile full \
    --out_base "$LOCAL_JOB_BASE/eval_${RUN_NAME}_stage2"

echo "##### END ##### (wandb run: $RUN_ID)"
