#!/bin/bash
#SBATCH --job-name=eval_long
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/eval_long_%x_%j.out
exec 2>&1

# Long-profile evals handed off from the B200 container, which could not fit
# them before its deadline. Every checkpoint here is on the hub, so this runs
# from a repo id with nothing copied between machines.
#
#   sbatch -J oursd003_s0 elsa/scripts/log_cluster/slurm_eval_long_handoff.sh oursd003_seed0
#
# run_name follows s3_8b_<arm>_s80_seed<N>, which long_tsv_results/
# harvest_long_tsv.py strips back to <arm> -- so these seeds merge into the
# arm's existing block instead of opening a second one-seed block.
set -u
ARM=${1:?"usage: sbatch slurm_eval_long_handoff.sh <oursd003_seed0|oursd003_seed1|alpsretrainnoopd_seed1|alpspgdtr_seed{0,1,42}>"}

# 8B s80 defaults; the 4B arms below override them.
PROJECT=reasoning_qwen3_8b_nostrip8192
SPARSITY=0.8
# Set by a repair arm to log into an existing wandb run / run one benchmark.
RESUME_ID=
BENCHMARKS=

case "$ARM" in
  # 'Ours (delta=0.03)'. Seed 42 is the training run's own eval (wandb
  # keuegrrb, pushed this repo at 2026-09-18T00:09 -- same checkpoint).
  oursd003_seed0)
    MODEL=cosmos1030/gmp-kd3e-1-8b-s80pct-lr1e-4_20260918_090751
    RUN=s3_8b_oursd003_s80_seed0; SEED=0 ;;
  oursd003_seed1)
    MODEL=cosmos1030/gmp-kd3e-1-8b-s80pct-lr1e-4_20260918_090751
    RUN=s3_8b_oursd003_s80_seed1; SEED=1 ;;
  # 'ALPS+retrain w/o OPD'. Seed 0 (wandb lk7ios48) ran from the B200-local
  # dir gmp_8b_s80pct_lr0.0001_20260917_043650_p392212; this repo carries that
  # path inside its own eval details, i.e. it is that checkpoint on the hub.
  alpsretrainnoopd_seed1)
    MODEL=cosmos1030/gmp-kd5e-1-8b-s80pct-lr1e-4_20260917_105733
    RUN=s3_8b_alpsretrainnoopd_s80_seed1; SEED=1 ;;
  # Repair arm. In job 51031 math500 died before generating anything: two jobs
  # started in the same second, both ran nltk.download into ~/nltk_data, and
  # the runner read punkt.zip while it was still being written
  # (zipfile.BadZipFile). The other four benchmarks ran fine once the download
  # completed. This re-runs math500 alone into the SAME wandb run so the seed
  # keeps one row. Submit it with --dependency=afterany:<that job> -- two
  # processes writing one run's summary would race.
  oursd003_seed0_math500)
    MODEL=cosmos1030/gmp-kd3e-1-8b-s80pct-lr1e-4_20260918_090751
    RUN=s3_8b_oursd003_s80_seed0; SEED=0
    RESUME_ID=xu8nbgav; BENCHMARKS=math500 ;;
  # 4B s70, ALPS mask + PGD with the trust region (klb 0.02) -- the B200
  # training finished 2026-09-21 21:27 KST but its built-in eval was killed to
  # free the GPU, so the checkpoint has no numbers at all. Its wandb TRAINING
  # run landed in the 8B project (launcher inherited the 8B default, fixed in
  # b9cfaee); this eval is logged to the 4B project on purpose, which is where
  # the arm belongs.
  alpspgdtr_seed0|alpspgdtr_seed1|alpspgdtr_seed42)
    MODEL=cosmos1030/gmp-4b-s70pct-lr0.0001-onpol-lmda0.33-20260921-212708-p265824
    SEED=${ARM#alpspgdtr_seed}
    RUN=s3_4b_alpspgdtr_s70_seed${SEED}
    PROJECT=reasoning_qwen3_4b_nostrip8192; SPARSITY=0.7 ;;
  *) echo "!! unknown arm '$ARM'" >&2; exit 1 ;;
esac

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

REPO=/home/doyoonkim/projects/onpolicyelsa_code
export HF_HOME=/home/shared/huggingface
export TOKENIZERS_PARALLELISM=false
# vllm 0.10 in rac runs V1; VLLM_USE_V1=0 is a B200 workaround that does not
# apply here (slurm_eval_lighteval_only.sh leaves it unset for the same reason).
export VLLM_USE_V1=1
export VLLM_HOST_IP=127.0.0.1
export TMPDIR=/tmp/${USER}/job_${SLURM_JOB_ID}
export WANDB_DIR=$TMPDIR
export TRITON_CACHE_DIR=$TMPDIR/triton
mkdir -p "$TMPDIR"

# lighteval writes thousands of small parquet files; keep that off NFS and copy
# only the parquet back at exit, so a job killed at its time limit still leaves
# the generations behind.
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
echo "  model $MODEL   seed $SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO/elsa"
# ifeval's instruction checks import nltk at module import, so EVERY benchmark
# dies if punkt is missing or half-written -- two jobs starting in the same
# second both downloaded it and the loser read a truncated zip, which is how
# job 51031 lost math500. flock serialises that; a no-op once the data is there.
flock /tmp/${USER}_nltk.lock \
    python -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt', 'punkt_tab')]" || true

EXTRA=()
[ -n "$RESUME_ID" ]  && EXTRA+=(--wandb_run_id "$RESUME_ID")
[ -n "$BENCHMARKS" ] && EXTRA+=(--benchmarks "$BENCHMARKS")

# $PROJECT/$SPARSITY, never a literal: a 4B arm logged into the 8B project
# with sparsity 0.8 is exactly the mix-up b9cfaee fixed on the training side.
python scripts/eval_full.py \
    --model_path "$MODEL" \
    --wandb_project "$PROJECT" \
    --wandb_entity dyk6208-gwangju-institute-of-science-and-technology \
    --run_name "$RUN" \
    --method gmp --sparsity "$SPARSITY" \
    --profile long --seeds "$SEED" \
    "${EXTRA[@]}" \
    --tp_size 1 --gpu_util 0.90 \
    --skip_ppl --skip_zeroshot \
    --out_base "$OUT_LOCAL"
echo "##### END ($?) #####"
