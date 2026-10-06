#!/bin/bash
#SBATCH --job-name=coverage_fr
# normal QOS only -- the hpgpu partitions are saturated by the long-profile
# eval sweep, and this needs one GPU for forward passes, no generation.
#SBATCH --partition=RTX6000ADA,A6000,L40S,A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

STATES=${1:-/home1/doyoonkim/projects/elsa/logs/policy_divergence/n30_k64_clean}
MODELS=${2:-dense,ours:s70,noopd55:s70,alps_sft:s70}
NROLL=${3:-8}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
OUTDIR=/home1/doyoonkim/projects/elsa/logs/policy_divergence/coverage
mkdir -p "$OUTDIR"

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
[ -z "${LOCAL_JOB_BASE:-}" ] && LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/slurm"
LOG="$OUTDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
LOCAL_LOG="$LOCAL_JOB_BASE/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
( while true; do cp "$LOCAL_LOG" "$LOG" 2>/dev/null || true; sleep 30; done ) &
M=$!
trap 'kill $M 2>/dev/null; cp "$LOCAL_LOG" "$LOG" 2>/dev/null || true' EXIT

export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false

echo "=== fixed vs rollout coverage ===  NODE=$(hostname) JOB=$SLURM_JOB_ID"
echo "states=$STATES  models=$MODELS  n_roll_per_prompt=$NROLL"
cd /home1/doyoonkim/projects/elsa
$PYTHON scripts/coverage_fixed_vs_rollout.py \
    --states_dir "$STATES" --models "$MODELS" \
    --n_roll "$NROLL" \
    --out "$OUTDIR/coverage_${SLURM_JOB_ID}.npz"
echo "=== EXIT: $? ==="
