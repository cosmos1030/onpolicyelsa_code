#!/bin/bash
#SBATCH --job-name=cot_models
# 80GB only: ten 4B models are loaded in sequence and a 40GB card leaves no
# margin if any one of them lingers a moment past its free.
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# The mirror of the policy-divergence figure: one fixed text (the dataset CoT)
# read by every model, so the representation moves instead of the text.
# Reuses the cached prompts and dense rollouts, so no dataset download.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
DENSE="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
SAMPLES=/home1/doyoonkim/projects/elsa/logs/policy_divergence/n6_k96_base

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
[ -z "${LOCAL_JOB_BASE:-}" ] && LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/slurm"

OUTDIR=/home1/doyoonkim/projects/elsa/logs/policy_divergence/cot_through_models
mkdir -p "$OUTDIR"
NFS_LOG="$OUTDIR/cot_models_${SLURM_JOB_ID}.out"
LOCAL_LOG="$LOCAL_JOB_BASE/slurm/cot_models_${SLURM_JOB_ID}.out"
( while true; do cp "$LOCAL_LOG" "$NFS_LOG" 2>/dev/null || true; sleep 30; done ) &
MIRROR=$!
trap 'kill $MIRROR 2>/dev/null; cp "$LOCAL_LOG" "$NFS_LOG" 2>/dev/null || true' EXIT

export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== CoT through each model ===  NODE=$(hostname) JOB=$SLURM_JOB_ID"

cd /home1/doyoonkim/projects/elsa
$PYTHON scripts/cot_through_models.py \
    --samples_dir "$SAMPLES" \
    --models dense=$DENSE \
             alps_s50=cosmos1030/alps-qwen3-4b-s50pct \
             alps_s60=cosmos1030/alps-qwen3-4b-s60pct \
             alps_s70=cosmos1030/alps-qwen3-4b-s70pct \
             sparsegpt_s50=cosmos1030/sparsegpt-qwen3-4b-s50pct \
             sparsegpt_s60=cosmos1030/sparsegpt-qwen3-4b-s60pct \
             sparsegpt_s70=cosmos1030/sparsegpt-qwen3-4b-s70pct \
             ours_s50=cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260903_142204 \
             ours_s60=cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260903_071754 \
             ours_s70=cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260901_080954 \
             noopd_s70=/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_20260913_132405 \
             noopd_s60=/home1/doyoonkim/projects/elsa/models/gmp_s60pct_lr0.0001_20260913_124306 \
             noopd_s50=/home1/doyoonkim/projects/elsa/models/gmp_s50pct_lr0.0001_20260913_062719 \
    --layers 18 36 \
    --seg_len 256 --n_seg 8 \
    --outdir "$OUTDIR"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
