#!/bin/bash
#SBATCH --job-name=onpol_mismatch
# Two partitions, because A100-80GB routinely sits 170+ jobs deep while 4A100
# sits under 10. hpgpu is the one QOS both accept, so the job can take whichever
# frees first. A 4B model peaks near 22GB here, so a 40GB card is plenty.
#SBATCH --partition=A100-80GB,4A100
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
# Measured runs take 16-19 min. An oversized wall is not free: the backfill
# scheduler can only slot a job into a gap it fits in, so a 12h request on a
# 20min job waits for a full-size opening instead of taking the next crack.
#SBATCH --time=01:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# On-policy mismatch diagnostic (see onpolicy_mismatch_diag.py).
#
# Measures KL(P||Q) for one ALPS-pruned Qwen3-4B under two context
# distributions -- the dense model's rollouts vs the pruned model's own -- on
# the SAME MATH-500 prompts. Run once per sparsity; the point of interest is
# whether D_self - D_dense grows across 50 -> 60 -> 70%.
#
# ALPS on purpose: it is the one-shot baseline and has seen no on-policy
# signal, so it cannot have been shaped to reduce the very mismatch being
# measured. ALPS+recovery is excluded (its recovery includes OPD), and
# SparseGPT at 70% is degraded far enough that any gap reads as "the model is
# broken" rather than as compounding drift.
#
# QOS must match the partition: A100-80GB takes hpgpu, not normal.
#
# Usage: sbatch slurm_onpolicy_mismatch_diag.sh <SPARSITY_PCT> [N_PROMPTS] [MAX_NEW] [SOURCE]
#   SOURCE=ot3      prompts + teacher CoT from an OpenThoughts3 slice held out
#                   from the ALPS calibration pool. The reference continuation
#                   is model-generated, <think>-prefixed and thousands of
#                   tokens -- the same kind of text as the rollouts, and the
#                   distribution off-policy KD actually trains on.
#   SOURCE=math500  prompts from MATH-500. Its reference is the human-written
#                   solution (~530 chars, contains Asymptote figure source), so
#                   only D_dense vs D_self are a like-for-like contrast there.

SPARSITY_PCT=${1:?"Usage: <SPARSITY_PCT: 50|60|70> [N_PROMPTS] [MAX_NEW]"}
N_PROMPTS=${2:-20}
MAX_NEW=${3:-8192}
SOURCE=${4:-math500}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
DENSE="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
PRUNED="cosmos1030/alps-qwen3-4b-s${SPARSITY_PCT}pct"

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/slurm"
OUTDIR=/home1/doyoonkim/projects/elsa/logs/onpol_mismatch
mkdir -p "$OUTDIR"

NFS_LOG="$OUTDIR/onpol_mismatch_s${SPARSITY_PCT}_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/onpol_mismatch_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT

export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
# MATH-500 is already in the HF cache; the pruned model is pulled from the Hub
# on first use, so this job needs network.
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
# No vLLM CuMemAllocator here (the engine is built and torn down plainly, no
# sleep mode), so expandable_segments is safe and is what keeps the vocab-sized
# log_softmax chunks from fragmenting.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1

echo "=== on-policy mismatch diag: ALPS Qwen3-4B s${SPARSITY_PCT}%, source=${SOURCE}, n_prompts=${N_PROMPTS}, max_new=${MAX_NEW} ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/onpolicy_mismatch_diag.py \
    --dense_model "$DENSE" \
    --pruned_model "$PRUNED" \
    --label "alps_s${SPARSITY_PCT}_${SOURCE}" \
    --n_prompts ${N_PROMPTS} \
    --prompt_source ${SOURCE} \
    --max_new_tokens ${MAX_NEW} \
    --out "$OUTDIR/alps_s${SPARSITY_PCT}_${SOURCE}_n${N_PROMPTS}_${SLURM_JOB_ID}.json"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
