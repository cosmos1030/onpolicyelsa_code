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
# core = the sparsities the figure plots (50/60/70) across all four methods.
# all  = adds the s30/s40 one-shot points, which exist only for ALPS/SparseGPT.
MODELSET=${5:-all}

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
LOCAL_LOG="$LOCAL_JOB_BASE/slurm/pol_diverge_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_LOG" "$NFS_LOG" 2>/dev/null || true' EXIT
# Copy on exit is not enough: this job runs for hours and its log lives on
# node-local storage the login node cannot read, so a run that is merely slow
# looks identical to one that is stuck. Mirror it every 30s as well.
( while true; do cp "$LOCAL_LOG" "$NFS_LOG" 2>/dev/null || true; sleep 30; done ) &
LOG_MIRROR_PID=$!
trap 'kill $LOG_MIRROR_PID 2>/dev/null; cp "$LOCAL_LOG" "$NFS_LOG" 2>/dev/null || true' EXIT

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
# One-shot baselines, our method, and ALPS followed by sparse SFT. That last one
# is the control that matters: our method trains and the one-shot baselines do
# not, so without it the figure would show "training keeps you near dense"
# rather than anything about our method. On avg5 ours beats ALPS->SFT by
# +0.86 / +2.99 / +4.72 at s50/s60/s70 -- a gap that widens with sparsity, which
# is the ordering the picture should reproduce if it is measuring anything real.
MODELS=(
  ours:s50=cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260903_142204
  ours:s60=cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260903_071754
  ours:s70=cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260901_080954
  alps_sft:s50=cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260812_132642
  alps_sft:s60=cosmos1030/gmp-kd3e-1-s60pct-lr1e-4_20260814_193400
  alps_sft:s70=cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260814_035030
  alps:s50=cosmos1030/alps-qwen3-4b-s50pct
  alps:s60=cosmos1030/alps-qwen3-4b-s60pct
  alps:s70=cosmos1030/alps-qwen3-4b-s70pct
  sparsegpt:s50=cosmos1030/sparsegpt-qwen3-4b-s50pct
  sparsegpt:s60=cosmos1030/sparsegpt-qwen3-4b-s60pct
  sparsegpt:s70=cosmos1030/sparsegpt-qwen3-4b-s70pct
)
if [ "$MODELSET" = "all" ]; then
  MODELS+=(
    alps:s30=cosmos1030/alps-s30pct_20260911_120849
    alps:s40=cosmos1030/alps-s40pct_20260911_120837
    sparsegpt:s30=cosmos1030/sparsegpt-s30pct_20260911_103959
    sparsegpt:s40=cosmos1030/sparsegpt-s40pct_20260911_103940
  )
fi

# NOTE the ALPS->SFT checkpoints share the gmp-kd3e-1- prefix with ours and are
# told apart only by date and wandb id: ours are 2026-09 (kp4bd255 / 4h0qqsze /
# r5j1uw8d), ALPS->SFT are 2026-08 (miysnxlq / dqocd3f8 / 5x4prktp). Picked by
# avg5 over the five reasoning tasks, not math500 alone.
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
    --with_teacher \
    --outdir "$OUTDIR"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
