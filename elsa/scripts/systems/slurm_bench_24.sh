#!/bin/bash
#SBATCH --job-name=bench24
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# Microbenchmark only -- minutes, not hours. An oversized wall just makes the
# backfill scheduler wait for an oversized hole.
#SBATCH --time=00:40:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# 2:4 semi-structured execution vs dense execution of the SAME weights.
#
# A100 80GB + BF16 on purpose: that is the pairing PyTorch's semi-structured
# path is documented and validated on, and it is the card the rest of the 4B
# work runs on. The point of this job is the control -- identical weights, two
# kernels -- so that "2:4 execution is X times faster" can be stated without
# generation length or accuracy entering the comparison.
#
# Usage: sbatch slurm_bench_24.sh [MODEL_DIR] [ITERS]

MODEL=${1:-/home1/doyoonkim/projects/elsa/models/scout_4b_2to4_best}
ITERS=${2:-50}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
ROOT=/home1/doyoonkim/projects
OUTDIR=$ROOT/elsa/logs/systems
mkdir -p "$OUTDIR"

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/slurm"
NFS_LOG="$OUTDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT

export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== 2:4 kernel microbenchmark: $MODEL ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$ROOT/elsa"

echo
echo "### step 1 -- is the mask hardware-valid 2:4?"
$PYTHON scripts/systems/check_24_validity.py "$MODEL" --limit 14
VALID=$?
if [ $VALID -ne 0 ]; then
    echo "mask is not loadable by the 2:4 kernel; stopping before timing."
    echo "##### END #####"
    exit $VALID
fi

echo
echo "### step 2 -- dense kernel vs 2:4 kernel, same weights"
$PYTHON scripts/systems/bench_linear_24.py \
    --model "$MODEL" --iters "$ITERS" \
    --out "$OUTDIR/linear_24_${SLURM_JOB_ID}.json"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
