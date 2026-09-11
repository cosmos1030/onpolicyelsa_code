#!/bin/bash
#SBATCH --job-name=pol_diverge
#SBATCH --partition=A100-80GB,4A100
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# One prompt, many rollouts per model, all encoded by the dense model.
# See policy_divergence_tsne.py for why this layout beats sampling many prompts.
#
# Usage: sbatch slurm_policy_divergence.sh [N_PROMPTS] [N_SAMPLES] [MAX_NEW] [TAG]

N_PROMPTS=${1:-6}
N_SAMPLES=${2:-96}
MAX_NEW=${3:-2048}
TAG=${4:-base}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
DENSE="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/slurm"

OUTROOT=/home1/doyoonkim/projects/elsa/logs/policy_divergence
# Not job-scoped: sampling is most of the runtime and the script caches it here.
OUTDIR="$OUTROOT/n${N_PROMPTS}_k${N_SAMPLES}_${TAG}"
mkdir -p "$OUTDIR"

NFS_LOG="$OUTROOT/pol_diverge_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/pol_diverge_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT

export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1

# Extra pruned models can be appended as a 5th argument onwards, same format.
MODELS=(
  alps:s50=cosmos1030/alps-qwen3-4b-s50pct
  alps:s60=cosmos1030/alps-qwen3-4b-s60pct
  alps:s70=cosmos1030/alps-qwen3-4b-s70pct
  sparsegpt:s50=cosmos1030/sparsegpt-qwen3-4b-s50pct
  sparsegpt:s60=cosmos1030/sparsegpt-qwen3-4b-s60pct
  sparsegpt:s70=cosmos1030/sparsegpt-qwen3-4b-s70pct
)
shift 4 2>/dev/null || shift $# 
if [ $# -gt 0 ]; then MODELS+=("$@"); fi

echo "=== policy divergence: $N_PROMPTS prompts x $N_SAMPLES samples ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  OUTDIR=$OUTDIR"
echo "MODELS: ${MODELS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa
$PYTHON scripts/policy_divergence_tsne.py \
    --dense_model "$DENSE" \
    --models "${MODELS[@]}" \
    --n_prompts ${N_PROMPTS} \
    --n_samples ${N_SAMPLES} \
    --max_new_tokens ${MAX_NEW} \
    --layers 18 36 \
    --outdir "$OUTDIR"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
