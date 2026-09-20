#!/bin/bash
# Seeds 0 and 1 for the two s70 arms the OPD ablation rests on.
#
# Why: the OPD claim is SCOUT 57.25 vs SCOUT-w/o-OPD 56.54, i.e. +0.71 avg5 --
# and both numbers are ONE seed (42). The dense 3-seed avg5 std is 0.27, but
# dense truncates 2-4% where these two truncate 14-53%, so that std is not a
# usable proxy here. Without seeds 0/1 the +0.71 cannot be called a result.
#
# Usage: bash eval_8b_opd_seeds.sh <ours|noopd>
set -u

ROOT=${ELSA_ROOT:-/NHNHOME/log-postech/doyoonkim}
REPO=${ELSA_REPO:-$ROOT/onpolicyelsa_code}
PYTHON=${PYTHON:-$ROOT/miniconda3/envs/rac/bin/python}
OUT_ROOT=${OUT_ROOT:-$ROOT/logs/eval_8b_long}
SEEDS=${SEEDS:-0,1}
PROFILE=${PROFILE:-long}
GPU_UTIL=${GPU_UTIL:-0.90}
TP_SIZE=${TP_SIZE:-1}
WANDB_PROJECT=${WANDB_PROJECT:-reasoning_qwen3_8b_nostrip8192}

die () { echo "!! $*" >&2; exit 1; }
ARM=${1:?"usage: eval_8b_opd_seeds.sh <ours|noopd>"}

case "$ARM" in
  # Same checkpoint s3_8b_ours_s70 (3y9jvl5s) used for seed 42. Already in the
  # HF cache here at 16GB, so the repo id costs nothing.
  ours)  M=cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260911_015957
         RUN=s3_8b_ours_s70_seeds01 ;;
  # The 2-term (NTP+KD, gmp_onpolicy_kd_lambda=0) s70 run g5htdr2q. Its seed-42
  # numbers came from the TRAINING job's inline eval, not an s3_* run -- hence
  # this script rather than eval_8b_long.sh. Path from that job's
  # "Saved pruned model to" line; the dir name has no onpol_lmda, matching OPD off.
  noopd) M=$ROOT/models/gmp_8b_s70pct_lr0.0001_20260917_150541_p886909
         RUN=s3_8b_ours_s70_noopd_seeds01 ;;
  *) die "unknown arm '$ARM' (ours | noopd)" ;;
esac

case "$M" in
  /*) [ -f "$M/config.json" ] || die "no config.json under $M"
      [ "$(ls "$M"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ] || die "no safetensors under $M" ;;
esac

MARK="$OUT_ROOT/.done_${RUN}"
if [ -f "$MARK" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "== $ARM already done ($MARK) -- skipping. FORCE=1 to redo."; exit 0
fi

source "$ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate rac
[ -x "$PYTHON" ] || die "no python at $PYTHON"
grep -q "_LONG_BENCHMARKS" "$REPO/elsa/lib/lighteval_bench.py" || die "checkout predates the long profile"
grep -q "_patch_ifeval_json" "$REPO/elsa/scripts/lighteval_patched_runner.py" \
  || die "checkout lacks the ifeval RecursionError patch -- s70 models trip it"

export HF_TOKEN=${HF_TOKEN:-$(cat "$ROOT/secrets/hf_token" 2>/dev/null)}
export WANDB_API_KEY=${WANDB_API_KEY:-$(cat "$ROOT/secrets/wandb_api_key" 2>/dev/null)}
[ -n "${WANDB_API_KEY:-}" ] || die "no WANDB_API_KEY"
export HF_HOME=$ROOT/.cache/huggingface
export VLLM_CACHE_ROOT=$ROOT/.cache/vllm
export TRITON_CACHE_DIR=$ROOT/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$ROOT/.cache/torchinductor
export TMPDIR=/tmp TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_HOST_IP=127.0.0.1

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/${RUN}_$(date +%Y%m%d_%H%M%S).log"
echo "=== 8B OPD-ablation extra seeds ==="
echo "  arm $ARM   model $M"
echo "  run $RUN -> $WANDB_PROJECT   seeds $SEEDS   profile $PROFILE"
echo "  gpu ${CUDA_VISIBLE_DEVICES:-all}   log $LOG"

cd "$REPO/elsa"
"$PYTHON" scripts/eval_full.py \
    --model_path "$M" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN" \
    --method gmp --sparsity 0.7 \
    --profile "$PROFILE" \
    --seeds "$SEEDS" \
    --tp_size "$TP_SIZE" --gpu_util "$GPU_UTIL" \
    --skip_ppl --skip_zeroshot \
    --out_base "$OUT_ROOT/eval_${RUN}" 2>&1 | tee "$LOG"

CODE=${PIPESTATUS[0]}
if [ "$CODE" -eq 0 ] || grep -q "lighteval bench done\|_mean" "$LOG"; then touch "$MARK"; fi
echo "=== EXIT: $CODE ===  log: $LOG"
exit $CODE
