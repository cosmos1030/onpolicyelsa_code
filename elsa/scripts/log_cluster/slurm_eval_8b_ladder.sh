#!/bin/bash
#SBATCH --job-name=eval8b
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/eval8b_%x_%j.out
exec 2>&1

# The 8B s50/s60 ladder at seeds 0 and 1 -- seed 42 already exists for all
# eight arms, so this is what turns a one-seed row into a three-seed one.
#
#   sbatch -J ours_s50 elsa/scripts/log_cluster/slurm_eval_8b_ladder.sh ours_s50
#   SEEDS=0 sbatch ... ours_s50        # one seed per job instead of both
#
# Run name is s3_8b_<arm>_seeds01, matching b200_scripts/eval_8b_long.sh's
# RUN_SUFFIX convention: harvest_long_tsv.py strips _seeds01 and merges these
# seeds into the arm's existing seed-42 block. A run named s3_8b_<arm> would
# instead collide with the seed-42 run's name and split the arm in two.
set -u
ARM=${1:?"usage: sbatch slurm_eval_8b_ladder.sh <{ours,alpsretrain,alps,sparsegpt}_s{50,60}>"}
SEEDS=${SEEDS:-0,1}

case "$ARM" in
  ours_s50)        REPO_ID=cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260907_210125;  ME=gmp;       SP=0.5 ;;
  alpsretrain_s50) REPO_ID=cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260908_053513;  ME=gmp;       SP=0.5 ;;
  # The seed-42 runs used the B200-local dirs models/qwen3_8b_alps_s{50,60}pct;
  # from here the same weights are only reachable as these hub repos. ALPS
  # pruning is deterministic given the dense model and calibration set, but if
  # the three seeds ever disagree oddly, check that assumption first.
  alps_s50)        REPO_ID=cosmos1030/qwen3-8b-alps-s50pct;                      ME=alps;      SP=0.5 ;;
  sparsegpt_s50)   REPO_ID=cosmos1030/qwen3-8b-sgpt-s50pct-ot80fw20;             ME=sparsegpt; SP=0.5 ;;
  ours_s60)        REPO_ID=cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260907_142152;  ME=gmp;       SP=0.6 ;;
  alpsretrain_s60) REPO_ID=cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260908_054305;  ME=gmp;       SP=0.6 ;;
  alps_s60)        REPO_ID=cosmos1030/qwen3-8b-alps-s60pct;                      ME=alps;      SP=0.6 ;;
  sparsegpt_s60)   REPO_ID=cosmos1030/qwen3-8b-sgpt-s60pct-ot80fw20;             ME=sparsegpt; SP=0.6 ;;
  *) echo "!! unknown arm '$ARM'" >&2; exit 1 ;;
esac

# Prefer a plain local copy fetched with curl. huggingface_hub's cache lives on
# NFS here and its per-blob locks are not cleaned up when a job is killed: on
# 2026-09-21 four jobs sat 11h at "loading model weights" with the GPU at 0%,
# waiting on a lock left by a job killed the night before. curl does not lock.
# .download_complete is written ONLY after every shard has arrived. Checking
# for config.json + any .safetensors is not enough: an interrupted download
# left 30MB of shard 1 of 4 in models/ours_s50, this script took it for a
# finished copy, and all five benchmarks died instantly on the missing
# tokenizer -- the job then "succeeded" in four minutes with nothing in it.
LOCAL=/home/doyoonkim/models/$ARM
if [ -f "$LOCAL/.download_complete" ]; then
  MODEL=$LOCAL
else
  MODEL=$REPO_ID
  echo "(warning) $LOCAL not populated -- falling back to the hub, which can wedge"
fi

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

PROJECT_FOR_CHECK=reasoning_qwen3_8b_nostrip8192
RUN=s3_8b_${ARM}_seeds01
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
echo "  model $MODEL   method $ME   sparsity $SP   seeds $SEEDS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO/elsa"
# ifeval imports nltk at module import, so a missing or half-written punkt
# kills every benchmark, not just ifeval.
flock /tmp/${USER}_nltk.lock \
    python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab')]" || true

# Another box may already be on this checkpoint: several servers share this
# wandb project and hub, and a duplicate burns a GPU for hours for nothing.
if ! python "$REPO/elsa/scripts/log_cluster/preflight_dup_check.py" \
        --model "$MODEL" --seeds "$SEEDS" --run "$RUN" --project "$PROJECT_FOR_CHECK"; then
    echo "##### SKIPPED (duplicate running elsewhere) #####"
    exit 0
fi

python scripts/eval_full.py \
    --model_path "$MODEL" \
    --wandb_project reasoning_qwen3_8b_nostrip8192 \
    --wandb_entity dyk6208-gwangju-institute-of-science-and-technology \
    --run_name "$RUN" \
    --method "$ME" --sparsity "$SP" \
    --profile long --seeds "$SEEDS" \
    --tp_size 1 --gpu_util 0.90 \
    --skip_ppl --skip_zeroshot \
    --out_base "$OUT_LOCAL"
echo "##### END ($?) #####"
