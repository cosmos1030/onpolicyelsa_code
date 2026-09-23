#!/bin/bash
#SBATCH --job-name=cov_sweep
# One GPU, forward passes only (no generation): the rollouts already exist in
# the pool. Same partitions as the single-layer version -- normal QOS so this
# does not queue behind the hpgpu eval sweep.
#SBATCH --partition=RTX6000ADA,A6000,L40S,A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
# The single-layer run wrote its npz in ~10 min of GPU time; extracting every
# layer from the SAME forward costs nothing extra on the GPU, only host-side
# pooling and a larger npz, so 6h is generous.
#SBATCH --time=06:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

STATES=${1:-/home1/doyoonkim/projects/elsa/logs/policy_divergence/n30_k64_clean}
MODELS=${2:-dense,ours:s70,norefresh:s70,noopd55:s70,alps_sft:s70}
NROLL=${3:-8}
LAYERS=${4:-"4 9 13 18 22 27 31 36"}
# 인코더(teacher). 1.7B arm을 재려면 1.7B dense를 줘야 한다 -- 4B 인코더로 1.7B
# 롤아웃을 읽으면 크기 차이를 재게 된다. 기본값은 스크립트 쪽 4B.
DENSE=${5:-}
DENSE_ARG=""
[ -n "$DENSE" ] && DENSE_ARG="--dense $DENSE"

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
OUTDIR=/home1/doyoonkim/projects/elsa/logs/policy_divergence/coverage
mkdir -p "$OUTDIR"

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
[ -z "${LOCAL_JOB_BASE:-}" ] && LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/slurm"
# Follow the job NAME (--output uses %x) and mirror while running, or a crash
# leaves nothing on NFS.
LOG="$OUTDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
LOCAL_LOG="$LOCAL_JOB_BASE/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
( while true; do cp "$LOCAL_LOG" "$LOG" 2>/dev/null || true; sleep 30; done ) &
M=$!
trap 'kill $M 2>/dev/null; cp "$LOCAL_LOG" "$LOG" 2>/dev/null || true' EXIT

export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== probe depth sweep ===  NODE=$(hostname) JOB=$SLURM_JOB_ID"
echo "states=$STATES  models=$MODELS  n_roll=$NROLL  layers=$LAYERS  dense=${DENSE:-기본(4B)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa
$PYTHON scripts/coverage_layer_sweep.py \
    --states_dir "$STATES" --models "$MODELS" \
    --n_roll "$NROLL" --layers $LAYERS $DENSE_ARG \
    --out "$OUTDIR/sweep_${SLURM_JOB_ID}.npz"
EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
