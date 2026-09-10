#!/bin/bash
#SBATCH --job-name=hstate_drift
# A100-80GB routinely sits 170+ jobs deep while 4A100 sits under 10, and hpgpu
# is the one QOS both accept. A 4B model here peaks well under 40GB.
#SBATCH --partition=A100-80GB,4A100
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# Hidden-state drift: do the states a model visits while generating freely pull
# away from the states a fixed reasoning trace induces, as pruning gets harder?
# See hidden_state_drift.py for the three confounds this is built around
# (token depth, text identity, and probe leakage).
#
# The pruned models are pulled from the Hub, so this job needs network -- hence
# the internet exclude list rather than the plain one.
#
# Usage: sbatch slurm_hidden_state_drift.sh [N_PROMPTS] [MAX_NEW] [SOURCE]

N_PROMPTS=${1:-50}
MAX_NEW=${2:-3300}
SOURCE=${3:-ot3}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
DENSE="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/slurm"

OUTROOT=/home1/doyoonkim/projects/elsa/logs/hstate_drift
# Deliberately not job-scoped: rollout generation is ~12 min of the run and the
# script caches it here, so a job that dies during encoding resumes cheaply.
OUTDIR="$OUTROOT/${SOURCE}_n${N_PROMPTS}"
mkdir -p "$OUTDIR"

NFS_LOG="$OUTROOT/hstate_drift_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/hstate_drift_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT

export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
# No vLLM sleep mode here (engines are built and torn down plainly), so
# expandable_segments is safe and keeps the long-sequence forwards from
# fragmenting the allocator.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1

echo "=== hidden-state drift: Qwen3-4B dense vs ALPS s50/s60/s70 ==="
echo "source=${SOURCE} n_prompts=${N_PROMPTS} max_new=${MAX_NEW}"
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  OUTDIR=$OUTDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/hidden_state_drift.py \
    --dense_model "$DENSE" \
    --pruned_models s50=cosmos1030/alps-qwen3-4b-s50pct \
                    s60=cosmos1030/alps-qwen3-4b-s60pct \
                    s70=cosmos1030/alps-qwen3-4b-s70pct \
    --n_prompts ${N_PROMPTS} \
    --max_new_tokens ${MAX_NEW} \
    --prompt_source ${SOURCE} \
    --layers 18 36 \
    --outdir "$OUTDIR"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
