#!/bin/bash
#SBATCH --job-name=drift_vis
# Pure CPU: nothing here touches a GPU.
#SBATCH --partition=cpu-max16
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# One non-t-SNE view of the saved fixed-vs-self hidden states.
# Usage: sbatch slurm_drift_visual.sh <STATES_DIR> <VARIANT> [LAYER] [WINDOW]

STATES_DIR=${1:?"usage: <STATES_DIR> <VARIANT> [LAYER] [WINDOW]"}
VARIANT=${2:?"variant name"}
LAYER=${3:-18}
WINDOW=${4:-grid}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/slurm"

LOGDIR=/home1/doyoonkim/projects/elsa/logs/hstate_drift
NFS_LOG="$LOGDIR/drift_vis_${VARIANT}_L${LAYER}_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/drift_vis_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT

export TMPDIR=/tmp
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "=== drift visual: $VARIANT  layer=$LAYER  window=$WINDOW ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  STATES=$STATES_DIR"

cd /home1/doyoonkim/projects/elsa
$PYTHON scripts/drift_visuals.py \
    --states_dir "$STATES_DIR" \
    --variant "$VARIANT" \
    --layer ${LAYER} \
    --window ${WINDOW}

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
