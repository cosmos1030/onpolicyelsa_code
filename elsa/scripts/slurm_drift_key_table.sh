#!/bin/bash
#SBATCH --job-name=drift_tbl
#SBATCH --partition=cpu-max16
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# The decisive table: raw / pooled-centered / seqmean separability, each with
# the dense-encoder control on identical tokens. See drift_key_table.py.
# Usage: sbatch slurm_drift_key_table.sh <STATES_DIR>

STATES_DIR=${1:?"usage: <STATES_DIR>"}
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/slurm"

LOGDIR=/home1/doyoonkim/projects/elsa/logs/hstate_drift
NFS_LOG="$LOGDIR/drift_tbl_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/drift_tbl_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT

export TMPDIR=/tmp
export OMP_NUM_THREADS=8

echo "=== drift key table: $STATES_DIR ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"

cd /home1/doyoonkim/projects/elsa
$PYTHON scripts/drift_key_table.py --states_dir "$STATES_DIR"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
