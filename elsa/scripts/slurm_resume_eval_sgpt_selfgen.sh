#!/bin/bash
#SBATCH --job-name=resume_eval_sgpt
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# Re-run ONLY the lighteval bench for an already-pruned SparseGPT self-gen
# checkpoint, logging into the SAME wandb run the pruning job created.
#
# Why this exists: prune_and_eval.py's order is prune -> save+push -> ppl ->
# zero-shot -> lighteval. The 2026-09-06 self-gen sweep (jobs 870723-870730)
# completed pruning and saved/pushed every checkpoint, then ran out of its 12h
# wall inside lighteval's generative tasks -- measured 127-191 s/it, i.e. 6-14h
# of ETA remaining with ~2h left. Nothing about the pruning needs redoing.
#
# lighteval caches generations under <model_dir>/<model_hash>/<task>/, keyed on
# the model-args string, so passing the SAME --gpu_util the original run used
# makes already-generated samples skip generation and only re-score. Partial
# progress from the timed-out run is therefore not wasted -- which is why those
# jobs were left to run to their wall instead of being cancelled early.
#
# The 24h limit (vs the original 12h) is deliberate: the original died BECAUSE
# 12h was not enough for lighteval alone, and this job does only that part.
#
# Usage:
#   sbatch --partition=<P> slurm_resume_eval_sgpt_selfgen.sh \
#       <MODEL_DIR> <WANDB_PROJECT> <WANDB_RUN_ID> <SPARSITY> [GPU_UTIL] [TP_SIZE]
# e.g.
#   sbatch --partition=RTX6000ADA slurm_resume_eval_sgpt_selfgen.sh \
#       /home1/doyoonkim/projects/elsa/models/qwen3_4b_sgpt_s70pct_n128_selfgenv3 \
#       reasoning_qwen3_4b uw9vc7du 0.7

MODEL_DIR=${1:?"Usage: <MODEL_DIR> <WANDB_PROJECT> <WANDB_RUN_ID> <SPARSITY> [GPU_UTIL] [TP_SIZE]"}
WANDB_PROJECT=${2:?"Usage: <MODEL_DIR> <WANDB_PROJECT> <WANDB_RUN_ID> <SPARSITY> [GPU_UTIL] [TP_SIZE]"}
WANDB_RUN_ID=${3:?"Usage: <MODEL_DIR> <WANDB_PROJECT> <WANDB_RUN_ID> <SPARSITY> [GPU_UTIL] [TP_SIZE]"}
SPARSITY=${4:?"Usage: <MODEL_DIR> <WANDB_PROJECT> <WANDB_RUN_ID> <SPARSITY> [GPU_UTIL] [TP_SIZE]"}
# Must match what the original run used or lighteval's cache key changes and
# every already-generated sample is regenerated from scratch.
GPU_UTIL=${5:-0.9368}
TP_SIZE=${6:-1}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
RESUME_PY=/home1/doyoonkim/projects/b200_scripts/resume_eval_lighteval.py

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: no config.json under $MODEL_DIR -- not a saved model dir."; exit 1
fi

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/slurm"
mkdir -p /home1/doyoonkim/projects/elsa/logs

# The original sweep wrote its logs only to /local-data and left no trap, so
# when those jobs ended their logs became unreachable. Don't repeat that.
NFS_LOG="/home1/doyoonkim/projects/elsa/logs/resume_eval_sgpt_${SLURM_JOB_ID}_last.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/resume_eval_sgpt_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT

export ELSA_PATH=/home1/doyoonkim/projects/elsa
export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== resume lighteval: $(basename $MODEL_DIR) -> wandb $WANDB_PROJECT/$WANDB_RUN_ID ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  GPU_UTIL=$GPU_UTIL  TP=$TP_SIZE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."; exit 1
fi

$PYTHON "$RESUME_PY" \
    --model_dir "$MODEL_DIR" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_id "$WANDB_RUN_ID" \
    --sparsity "$SPARSITY" \
    --gpu_util "$GPU_UTIL" \
    --tp_size "$TP_SIZE" \
    --profile quick \
    --no_hub

EXIT_CODE=$?
echo "=== resume_eval EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
