#!/bin/bash
#SBATCH --job-name=kl_fixed_vs_self
# A100-80GB is not in this list: it only accepts hpgpu/add_hpgpu/test, and a
# one-GPU 4B job belongs on the 48GB cards anyway. Mixing them put the job
# straight into 'QOS not permitted to use this partition'.
# Widened to every 'normal'-QOS partition that fits two 4B models in bf16
# (~16GB) plus a (chunk, 151936) fp32 block: 48GB A6000/6000ADA, 46GB L40S,
# 40GB A100-PCIe. Staying off hpgpu partitions matters -- that QOS is
# saturated by the long-profile eval sweep, so an hpgpu job would queue
# behind it while these slot in immediately.
#SBATCH --partition=RTX6000ADA,A6000,L40S,A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out

# Token-level KL(dense || sparse) on fixed text vs the model's own rollouts.
# Two 4B models resident at once (~16GB bf16) plus one (chunk, 151936) fp32
# block, so a 48GB card is ample; A100 is in the list only for queue depth.
#
# Usage: sbatch slurm_kl_fixed_vs_selfgen.sh [STATES_DIR] [N_ROLLOUTS] [MODELS]

STATES_DIR=${1:-/home1/doyoonkim/projects/elsa/logs/policy_divergence/n30_k64_core}
N_ROLLOUTS=${2:-16}
MODELS=${3:-}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
OUTDIR=/home1/doyoonkim/projects/elsa/logs/policy_divergence/kl_fixed_vs_selfgen
mkdir -p "$OUTDIR"

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
[ -z "${LOCAL_JOB_BASE:-}" ] && LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/slurm"
# Copy the node-local log to NFS as it grows; without this a crash takes the
# only record of why with it.
LOG="$OUTDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
LOCAL_LOG="$LOCAL_JOB_BASE/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
( while true; do cp "$LOCAL_LOG" "$LOG" 2>/dev/null || true; sleep 30; done ) &
M=$!
trap 'kill $M 2>/dev/null; cp "$LOCAL_LOG" "$LOG" 2>/dev/null || true' EXIT

export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== KL fixed vs self-gen ===  NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "states=$STATES_DIR  n_rollouts=$N_ROLLOUTS  models=${MODELS:-all}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa
# KLDENSE / KLMAP let this score a different model family (the 1.7B loss-term
# arms). The teacher must match the students: a 4B teacher against 1.7B
# students measures the size gap, not the ablation.
EXTRA=()
[ -n "${KLDENSE:-}" ] && EXTRA+=(--dense "$KLDENSE")
[ -n "${KLMAP:-}" ]   && EXTRA+=(--model_map "$KLMAP")
$PYTHON scripts/kl_fixed_vs_selfgen.py \
    --states_dir "$STATES_DIR" \
    --out "$OUTDIR/kl_${SLURM_JOB_ID}.json" \
    --n_rollouts "$N_ROLLOUTS" \
    --models "$MODELS" "${EXTRA[@]}"
EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
