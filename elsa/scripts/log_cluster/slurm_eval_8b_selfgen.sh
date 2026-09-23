#!/bin/bash
#SBATCH --job-name=sg8b
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/sg8b_%x_%j.out
exec 2>&1

# 8B RAC ("+RAC" in Figure 2, selfgen v3 here) at seeds 0 and 1; seed 42 is
# already on the hub side. This is the SLURM version of
# elsa/scripts/handoff_8b_selfgen_seeds01.sh, which assumes a six-GPU box with
# no scheduler. One bug in that script is fixed here: it passes
# --wandb_run_name, which eval_full.py does not define, so argparse would have
# killed all six runs at once. Its benchmark list is kept as-is -- gpqa is
# deliberately skipped for these arms. harvest therefore leaves avg5 blank for
# seeds 0/1 (it needs all five), while the four benchmarks it does have still
# get their own mean and std.
#
#   sbatch -J alps_selfgen_s50 elsa/scripts/log_cluster/slurm_eval_8b_selfgen.sh alps 50
set -u
METHOD=${1:?"usage: sbatch slurm_eval_8b_selfgen.sh <alps|sparsegpt> <50|60|70>"}
SP_PCT=${2:?"usage: sbatch slurm_eval_8b_selfgen.sh <alps|sparsegpt> <50|60|70>"}
SEEDS=${SEEDS:-0,1}

case "$METHOD" in
  alps)      REPO_ID=cosmos1030/alps-selfgenv3-qwen3-8b-s${SP_PCT}pct; RUN=s3_8b_s${SP_PCT}_alps_selfgen_s01 ;;
  sparsegpt) REPO_ID=cosmos1030/sgpt-selfgenv3-qwen3-8b-s${SP_PCT}pct; RUN=s3_8b_s${SP_PCT}_sgpt_selfgen_s01 ;;
  *) echo "!! method must be alps or sparsegpt" >&2; exit 1 ;;
esac
case "$SP_PCT" in 50|60|70) ;; *) echo "!! sparsity must be 50, 60 or 70" >&2; exit 1 ;; esac

LOCAL=/home/doyoonkim/models/selfgen_8b_${METHOD}_s${SP_PCT}
MODEL=$REPO_ID
[ -f "$LOCAL/.download_complete" ] && MODEL=$LOCAL

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

REPO=/home/doyoonkim/projects/onpolicyelsa_code
PROJECT=reasoning_qwen3_8b_nostrip8192
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
echo "  model $MODEL   method $METHOD   sparsity 0.${SP_PCT}   seeds $SEEDS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO/elsa"
flock /tmp/${USER}_nltk.lock \
    python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab')]" || true

if ! python "$REPO/elsa/scripts/log_cluster/preflight_dup_check.py" \
        --model "$MODEL" --seeds "$SEEDS" --run "$RUN" --project "$PROJECT"; then
    echo "##### SKIPPED (duplicate running elsewhere) #####"
    exit 0
fi

python scripts/eval_full.py \
    --model_path "$MODEL" \
    --wandb_project "$PROJECT" \
    --wandb_entity dyk6208-gwangju-institute-of-science-and-technology \
    --run_name "$RUN" \
    --method "$METHOD" --sparsity "0.${SP_PCT}" \
    --profile long --seeds "$SEEDS" \
    --tp_size 1 --gpu_util 0.90 \
    --skip_ppl --skip_zeroshot \
    --out_base "$OUT_LOCAL"
echo "##### END ($?) #####"
