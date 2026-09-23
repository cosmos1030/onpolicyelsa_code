#!/bin/bash
#SBATCH --job-name=pol_diverge
# 48GB cards: one 4B model is resident at a time (vLLM for sampling, then the
# dense encoder), so an 80GB A100 is wasted here and its queue is 310 deep.
#SBATCH --partition=RTX6000ADA,A6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
# 12h: the 30x64 core run (job 914983) took 5:41 with 12 models, and adding
# the three noopd55 checkpoints puts it near 7h before any queue jitter.
#SBATCH --time=12:00:00
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
# DENSE_OVERRIDE / MODELS_OVERRIDE let this run a different model family
# (e.g. the 1.7B loss-term arms) without forking the script.
DENSE="${DENSE_OVERRIDE:-/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"

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

# Follow the job NAME, not the literal "pol_diverge": --output above uses %x,
# so an `sbatch --job-name=` override writes the real log somewhere this mirror
# was not looking. Job 939992 ran 3+ hours with NOTHING reaching NFS because it
# was launched as --job-name=pol_clean_n30 -- the same bug already fixed once in
# slurm_add_model_to_pooled.sh. Fail loudly instead of silently if it drifts
# again.
NFS_LOG="$OUTROOT/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
LOCAL_LOG="$LOCAL_JOB_BASE/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_LOG" "$NFS_LOG" 2>/dev/null || true' EXIT
# Copy on exit is not enough: this job runs for hours and its log lives on
# node-local storage the login node cannot read, so a run that is merely slow
# looks identical to one that is stuck. Mirror it every 30s as well.
( n=0
  while true; do
      if cp "$LOCAL_LOG" "$NFS_LOG" 2>/dev/null; then n=0
      else
          n=$((n+1))
          # Two minutes of failed copies means the path is wrong, not that the
          # file is late. Say so in the job's own stdout, which IS being written.
          [ "$n" = 4 ] && echo "WARNING: log mirror cannot read $LOCAL_LOG -- NFS log will stay empty" >&2
      fi
      sleep 30
  done ) &
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
  # The w/o-OPD arm, 0.5/0.5/0. Generated here rather than bolted on with
  # add_model_to_pooled.py afterwards, so it shares the pool's prompts and
  # sample count by construction.
  noopd55:s50=/home1/doyoonkim/projects/elsa/models/gmp_s50pct_lr5e-05_20260915_115020_p3476415
  noopd55:s60=/home1/doyoonkim/projects/elsa/models/gmp_s60pct_lr5e-05_20260915_165740_p3530899
  noopd55:s70=/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_20260915_173630_p908241
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

echo "=== policy divergence: $N_PROMPTS prompts x $N_SAMPLES samples ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  OUTDIR=$OUTDIR"
if [ -n "${MODELS_OVERRIDE:-}" ]; then
    read -r -a MODELS <<< "$MODELS_OVERRIDE"
fi
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
