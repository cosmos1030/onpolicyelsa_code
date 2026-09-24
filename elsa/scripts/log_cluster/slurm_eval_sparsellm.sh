#!/bin/bash
#SBATCH --job-name=sllm17b
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/sllm17b_%x_%j.out
exec 2>&1

# SparseLLM on Qwen3-1.7B, the cyan series in Figure 2. The 1.7B row has only
# SparseGPT / ALPS / SCOUT drawn; SparseLLM is missing at every sparsity even
# though all three checkpoints have been on the hub since 2026-07-20.
#
#   sbatch -J sparsellm_s50 elsa/scripts/log_cluster/slurm_eval_sparsellm_17b.sh 50
#
# Three seeds in one job: 1.7B is small enough that splitting them across GPUs
# would cost more in queue time than it saves.
set -u
SIZE=${1:?"usage: sbatch slurm_eval_sparsellm.sh <1.7b|4b|8b> <50|60|70>"}
SP_PCT=${2:?"usage: sbatch slurm_eval_sparsellm.sh <1.7b|4b|8b> <50|60|70>"}
SEEDS=${SEEDS:-0,1,42}
case "$SP_PCT" in 50|60|70) ;; *) echo "!! sparsity must be 50, 60 or 70" >&2; exit 1 ;; esac

# 1.7B uses the named repos; 4B has none, so it uses the dated push of the same
# weights. wandb shows why the hub has several dated repos per sparsity: the
# pruning was done once into /home1/.../models/qwen3_<size>_sparsellm_s<NN>pct
# and re-evaluated in July, August and September, each eval re-pushing the
# identical weights under a new timestamp. The ids below are the hub_model_id
# of the runs whose siblings produced the s80 numbers already in the table.
case "${SIZE}_${SP_PCT}" in
  1.7b_50) REPO_ID=cosmos1030/sparsellm-qwen3-1.7b-s50pct ;;
  1.7b_60) REPO_ID=cosmos1030/sparsellm-qwen3-1.7b-s60pct ;;
  1.7b_70) REPO_ID=cosmos1030/sparsellm-qwen3-1.7b-s70pct ;;
  4b_50)   REPO_ID=cosmos1030/sparsellm-s50pct_20260831_204946 ;;
  4b_60)   REPO_ID=cosmos1030/sparsellm-s60pct_20260831_210736 ;;
  4b_70)   REPO_ID=cosmos1030/sparsellm-s70pct_20260831_211530 ;;
  8b_50)   REPO_ID=cosmos1030/sparsellm-s50pct_20260901_050129 ;;
  8b_60)   REPO_ID=cosmos1030/sparsellm-s60pct_20260901_062412 ;;
  8b_70)   REPO_ID=cosmos1030/sparsellm-s70pct_20260901_073704 ;;
  *) echo "!! unknown size '$SIZE'" >&2; exit 1 ;;
esac
LOCAL=/home/doyoonkim/models/sparsellm_${SIZE}_s${SP_PCT}
MODEL=$REPO_ID
[ -f "$LOCAL/.download_complete" ] && MODEL=$LOCAL

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

REPO=/home/doyoonkim/projects/onpolicyelsa_code
export HF_HOME=/home/shared/huggingface
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=30
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export VLLM_HOST_IP=127.0.0.1
export TMPDIR=/tmp/${USER}/job_${SLURM_JOB_ID}
export WANDB_DIR=$TMPDIR
export TRITON_CACHE_DIR=$TMPDIR/triton
mkdir -p "$TMPDIR"

# s3_1.7b_sparsellm_s<NN>: harvest strips the size prefix and the _sNN, landing
# on METHOD['sparsellm'] = 'SparseLLM', and reads the sparsity back off the same
# _sNN for the row it belongs to.
PROJECT_FOR_CHECK=reasoning_qwen3_${SIZE}_nostrip8192
RUN=s3_${SIZE}_sparsellm_s${SP_PCT}
OUT_LOCAL=$TMPDIR/eval_${RUN}
DETAILS=$HOME/elsa_eval_long/${RUN}_${SLURM_JOB_ID}
save_details () {
    [ -d "$OUT_LOCAL" ] || return 0
    mkdir -p "$DETAILS"
    (cd "$OUT_LOCAL" && find . -name "*.parquet" -print0 2>/dev/null |
        while IFS= read -r -d "" f; do
            mkdir -p "$DETAILS/$(dirname "$f")"
            cp -n "$f" "$DETAILS/$f" 2>/dev/null || true
        done)
    echo "[details] $(find "$DETAILS" -name '*.parquet' 2>/dev/null | wc -l) parquet -> $DETAILS"
    rm -rf "$TMPDIR"
}
trap save_details EXIT

echo "=== $RUN ==="
echo "  host $(hostname)  job $SLURM_JOB_ID  gpu ${CUDA_VISIBLE_DEVICES:-?}"
echo "  model $MODEL   sparsity 0.${SP_PCT}   seeds $SEEDS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO/elsa"
flock /tmp/${USER}_nltk.lock \
    python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab')]" || true

# Another box may already be on this checkpoint: several servers share this
# wandb project and hub, and a duplicate burns a GPU for hours for nothing.
# ALLOW_DUP=1 is for a deliberate split: when a 3-seed job is too slow, its
# remaining seeds are launched as their own jobs and the parent is cancelled
# the moment it finishes the seed it is on. The overlap is intentional and
# short, so the guard would only get in the way.
if [ "${ALLOW_DUP:-0}" = "1" ]; then
  echo "[preflight] ALLOW_DUP=1 -- skipping the duplicate check on purpose"
elif ! python "$REPO/elsa/scripts/log_cluster/preflight_dup_check.py" \
        --model "$MODEL" --seeds "$SEEDS" --run "$RUN" --project "$PROJECT_FOR_CHECK"; then
    echo "##### SKIPPED (duplicate running elsewhere) #####"
    exit 0
fi

python scripts/eval_full.py \
    --model_path "$MODEL" \
    --wandb_project reasoning_qwen3_${SIZE}_nostrip8192 \
    --wandb_entity dyk6208-gwangju-institute-of-science-and-technology \
    --run_name "$RUN" \
    --method sparsellm --sparsity "0.${SP_PCT}" \
    --profile long --seeds "$SEEDS" \
    --tp_size 1 --gpu_util 0.90 \
    --skip_ppl --skip_zeroshot \
    --out_base "$OUT_LOCAL"
echo "##### END ($?) #####"
