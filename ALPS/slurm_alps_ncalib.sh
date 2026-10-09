#!/bin/bash
#SBATCH --job-name=alps_ncalib
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=/local-data/user-data/%u/alps_ncalib_%j/slurm_%j.out
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
exec 2>&1
# Standard ALPS (fixed calibration only) with a configurable calibration size --
# the "more data" control for --rollout_refine, whose mix refine pass sees 2x
# the calibration tokens of standard ALPS.
# Usage: sbatch slurm_alps_ncalib.sh <1.7b|4b> <SPARSITY> <NSAMPLES> [DENSE_SELFGEN]
#   DENSE_SELFGEN>0 adds that many windows rolled out once from the dense model
#   (rollout-refine recipe) to the NSAMPLES fixed windows.
SIZE=${1:?"Usage: sbatch slurm_alps_ncalib.sh <1.7b|4b> <SPARSITY> <NSAMPLES>"}
SPARSITY=${2:?"Usage: sbatch slurm_alps_ncalib.sh <1.7b|4b> <SPARSITY> <NSAMPLES>"}
NSAMPLES=${3:?"Usage: sbatch slurm_alps_ncalib.sh <1.7b|4b> <SPARSITY> <NSAMPLES>"}
SELFGEN=${4:-0}
TAG="n${NSAMPLES}"; SG_FLAG=""
if [ "$SELFGEN" -gt 0 ]; then TAG="n${NSAMPLES}sg${SELFGEN}"; SG_FLAG="--dense_selfgen_windows ${SELFGEN}"; fi
export VLLM_USE_V1=0 VLLM_HOST_IP=127.0.0.1
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
case "$SIZE" in
    1.7b) MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" ;;
    4b)   MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" ;;
    *) echo "unknown size $SIZE"; exit 1 ;;
esac
DATA="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
SAVED_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_${SIZE}_alps_${TAG}_s${SPARSITY_PCT}pct"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/alps_ncalib_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE"
DEBUG_COPY_DIR="/home1/doyoonkim/projects/elsa/logs/alps_${TAG}_${SIZE}_s${SPARSITY_PCT}_${SLURM_JOB_ID}"
mkdir -p "$DEBUG_COPY_DIR"
copy_log_on_exit() {
    cp "$LOCAL_JOB_BASE/slurm_${SLURM_JOB_ID}.out" "$DEBUG_COPY_DIR/" 2>/dev/null || true
}
trap copy_log_on_exit EXIT

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

echo "=== ALPS Qwen3-${SIZE} s${SPARSITY_PCT}% calib=${TAG} (fixed ${NSAMPLES} + dense self-gen ${SELFGEN}) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/ALPS
$PYTHON qwen3_alps.py \
    "$MODEL" \
    ${SPARSITY} \
    --data_path "$DATA" \
    --nsamples ${NSAMPLES} \
    ${SG_FLAG} \
    --rho 300.0 \
    --seed 42 \
    --save "$SAVED_MODEL" \
    --push_to_hub \
    --hub_model_id "cosmos1030/alps-${TAG}-qwen3-${SIZE}-s${SPARSITY_PCT}pct"
EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
