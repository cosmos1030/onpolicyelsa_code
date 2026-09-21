#!/bin/bash
# One 8B checkpoint, long profile, 3 seeds, five reasoning benchmarks.
#
# The 8B row of the main table is empty -- wandb has s3_8b_dense and nothing
# else -- while 4B and 1.7B are filled in on the cluster. Running it there
# would need 12 jobs x 2 A100s and would evict both of those, so it lands here.
#
# Usage: bash eval_8b_long.sh <arm>
#   arm = {sparsegpt,alps,alpsretrain,ours}_s{50,60,70}
# Env: SEEDS (default 0,1,42), GPU_UTIL (0.90), PROFILE (long), FORCE=1 to
#      redo an arm whose done-marker already exists.
set -u

ROOT=${ELSA_ROOT:-/NHNHOME/log-postech/doyoonkim}
REPO=${ELSA_REPO:-$ROOT/onpolicyelsa_code}
PYTHON=${PYTHON:-$ROOT/miniconda3/envs/rac/bin/python}
OUT_ROOT=${OUT_ROOT:-$ROOT/logs/eval_8b_long}
SEEDS=${SEEDS:-0,1,42}
PROFILE=${PROFILE:-long}
# B200 is 180GB and the evals are sequence-starved, not memory-bound: on an
# 80GB A100 the same jobs sat at 52-72% GPU with 0.85 reserved. 8B bf16 is
# ~16GB and a 32k-token KV sequence is ~4.8GB, so 0.90 leaves room for ~30
# concurrent sequences. Drop it if vLLM reports a cache-block failure.
GPU_UTIL=${GPU_UTIL:-0.90}
TP_SIZE=${TP_SIZE:-1}
WANDB_PROJECT=${WANDB_PROJECT:-reasoning_qwen3_8b_nostrip8192}

die () { echo "!! $*" >&2; exit 1; }
ARM=${1:?"usage: eval_8b_long.sh <{sparsegpt,alps,alpsretrain,ours}_s{50,60,70}>"}

case "$ARM" in
  sparsegpt_s50)  M=cosmos1030/qwen3-8b-sgpt-s50pct-ot80fw20;             ME=sparsegpt; SP=0.5 ;;
  sparsegpt_s60)  M=cosmos1030/qwen3-8b-sgpt-s60pct-ot80fw20;             ME=sparsegpt; SP=0.6 ;;
  sparsegpt_s70)  M=cosmos1030/qwen3-8b-sgpt-s70pct-ot80fw20;             ME=sparsegpt; SP=0.7 ;;
  # Local, not the hub repo: these three are already on this box at 16GB each
  # under models/, so pulling the identical weights from HF again is pure
  # wall-clock. The other nine arms were trained on the cluster and exist here
  # only on the hub, so they keep their repo ids.
  alps_s50)       M=$ROOT/models/qwen3_8b_alps_s50pct;                    ME=alps;      SP=0.5 ;;
  alps_s60)       M=$ROOT/models/qwen3_8b_alps_s60pct;                    ME=alps;      SP=0.6 ;;
  alps_s70)       M=$ROOT/models/qwen3_8b_alps_s70pct;                    ME=alps;      SP=0.7 ;;
  # ALPS mask + NTP/KD/OPD retrain. The tok512 runs, not the August tok256
  # ones -- 4B and 1.7B both used opd_gen_len=512, and mixing the two inside
  # one row would compare training recipes, not sparsity.
  alpsretrain_s50) M=cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260908_053513; ME=gmp; SP=0.5 ;;
  alpsretrain_s60) M=cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260908_054305; ME=gmp; SP=0.6 ;;
  alpsretrain_s70) M=cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260908_132323; ME=gmp; SP=0.7 ;;
  # SCOUT. Picked from the 8B sweep the same way 4B and 1.7B were: best
  # in-training (quick-profile) average at that sparsity's standard lr --
  # 5e-5 for s50/s60, 1e-4 for s70. wandb runs roq1vwuw / mgf2ka9s / ajm5l60w.
  ours_s50)       M=cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260907_210125;  ME=gmp; SP=0.5 ;;
  ours_s60)       M=cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260907_142152;  ME=gmp; SP=0.6 ;;
  ours_s70)       M=cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260911_015957;  ME=gmp; SP=0.7 ;;
  # s80. Three arms that were only ever scored at seed 42, as the long eval
  # embedded in their own training run -- there is no hub copy, so these point
  # at the training output dirs on this box. Arm names are chosen so that
  # harvest's auto_label() strips "_s80"/"_seedN" and lands on a METHOD key
  # whose text equals the LABELS entry for the seed-42 run; name them
  # alpsretrain_s80 and the new seeds would form a second "ALPS+retrain" block
  # instead of merging with the one already there.
  # ALPS s80 was scored once, at seed 0, by whatever drove s3_8b_s80_alps on
  # the cluster; seeds 1 and 42 come from here. "alps_s80" is deliberate --
  # auto_label() strips the _s80 and lands on METHOD['alps'] = 'ALPS', the
  # same label the seed-0 run already carries, so all three seeds merge.
  alps_s80)             M=$ROOT/models/qwen3_8b_alps_s80pct;                                          ME=alps; SP=0.8 ;;
  alpsretrain033_s80)   M=$ROOT/models/gmp_8b_s80pct_lr0.0001_onpol_lmda0.33_20260917_054518_p394552; ME=gmp; SP=0.8 ;;
  alpsretrainnoopd_s80) M=$ROOT/models/gmp_8b_s80pct_lr0.0001_20260917_043650_p392212;                ME=gmp; SP=0.8 ;;
  oursd003_s80)         M=$ROOT/models/gmp_8b_s80pct_lr0.0001_onpol_lmda0.33_20260918_035901_p1215965; ME=gmp; SP=0.8 ;;
  *) die "unknown arm '$ARM'" ;;
esac

# RUN_SUFFIX lets a later seed batch land in its OWN wandb run instead of a
# second run with the same name: harvest_long_tsv.py keys blocks off the run
# name, so two same-named runs become two half-empty blocks. Seeds 0/1 are run
# as s3_8b_<arm>_seeds01 and merged into the seed-42 block at harvest time.
RUN="s3_8b_${ARM}${RUN_SUFFIX:-}"
MARK="$OUT_ROOT/.done_${RUN}"
if [ -f "$MARK" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "== $ARM already done ($MARK) -- skipping. FORCE=1 to redo."
  exit 0
fi

source "$ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate rac

[ -x "$PYTHON" ] || die "no python at $PYTHON"
[ -d "$REPO/elsa" ] || die "no elsa/ under $REPO"
# --profile long is accepted by argparse even on a checkout whose library has
# no long table, and then dies inside lighteval_bench. Check for the table.
grep -q "_LONG_BENCHMARKS" "$REPO/elsa/lib/lighteval_bench.py" \
  || die "this checkout predates the long profile -- git pull in $REPO"

export HF_TOKEN=${HF_TOKEN:-$(cat "$ROOT/secrets/hf_token" 2>/dev/null)}
export WANDB_API_KEY=${WANDB_API_KEY:-$(cat "$ROOT/secrets/wandb_api_key" 2>/dev/null)}
[ -n "${WANDB_API_KEY:-}" ] || die "no WANDB_API_KEY -- eval_full.py treats a failed wandb.init as fatal"
export HF_HOME=$ROOT/.cache/huggingface
export VLLM_CACHE_ROOT=$ROOT/.cache/vllm
export TRITON_CACHE_DIR=$ROOT/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$ROOT/.cache/torchinductor
export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
# Deliberately NOT setting TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE: this
# container has internet and lighteval fetches datasets that are not cached.

mkdir -p "$OUT_ROOT"
# Ask wandb first. Three machines evaluate these checkpoints and none of them
# sees the others' queues; on 2026-09-21 this box spent 4.8 GPU-hours on an
# eval n84 was already running. wandb_dup_check.py exits 0 when the same TSV
# cell (label, sparsity, seed) or the same checkpoint is finished at 5/5 long
# or running with a live heartbeat; 2 means it could not reach wandb, in which
# case run anyway -- a duplicate costs GPU time, a wrong skip loses a result.
if [ "${FORCE:-0}" != "1" ]; then
  for s in ${SEEDS//,/ }; do
    # not `$(... | tail -1)`: $? would then be tail's status, always 0
    DUP=$("$PYTHON" "$REPO/b200_scripts/wandb_dup_check.py" "$WANDB_PROJECT" "$RUN" "$s" "$M" 2>/dev/null)
    RC=$?; DUP=$(printf '%s\n' "$DUP" | tail -1)
    case $RC in
      0) echo "== SKIP $RUN seed $s: $DUP"; SKIPPED="${SKIPPED:-} $s" ;;
      2) echo "== dup check failed for seed $s ($DUP) -- running anyway" ;;
    esac
  done
  if [ -n "${SKIPPED:-}" ]; then
    KEEP=$(for s in ${SEEDS//,/ }; do case " $SKIPPED " in *" $s "*) ;; *) echo -n "$s,";; esac; done)
    SEEDS=${KEEP%,}
    [ -n "$SEEDS" ] || { echo "== every requested seed is already covered -- nothing to run"; exit 0; }
    echo "== running remaining seeds: $SEEDS"
  fi
fi
LOG="$OUT_ROOT/${RUN}_$(date +%Y%m%d_%H%M%S).log"
echo "=== 8B long eval ==="
echo "  arm $ARM   model $M"
echo "  run $RUN -> $WANDB_PROJECT   seeds $SEEDS   profile $PROFILE"
echo "  tp $TP_SIZE   gpu_util $GPU_UTIL   log $LOG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

cd "$REPO/elsa"
# PPL and zero-shot skipped: this table is reasoning-only, and the zero-shot
# suite needs datasets this container has not cached (see README).
"$PYTHON" scripts/eval_full.py \
    --model_path "$M" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN" \
    --method "$ME" --sparsity "$SP" \
    --profile "$PROFILE" \
    --seeds "$SEEDS" \
    --tp_size "$TP_SIZE" --gpu_util "$GPU_UTIL" \
    --skip_ppl --skip_zeroshot \
    --out_base "$OUT_ROOT/eval_${RUN}" 2>&1 | tee "$LOG"

CODE=${PIPESTATUS[0]}
# 134/139 on teardown is a known benign GC abort: the results are already in
# wandb by then. Treat the run as done if the log says the benchmarks finished.
if [ "$CODE" -eq 0 ] || grep -q "lighteval bench done\|_mean" "$LOG"; then
  touch "$MARK"
fi
echo "=== EXIT: $CODE ===  log: $LOG"
exit $CODE
