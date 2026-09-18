#!/bin/bash
# One 4B s80 checkpoint, long profile, 3 seeds, 5 reasoning benchmarks.
#
# Why this exists: the recorded SCOUT-s80 numbers came from the TRAINING job's
# inline eval, which runs the 'quick' profile -- math500/gpqa/ifeval/lcb at
# 8192 with max_model_length ALSO 8192 (the prompt eats the generation budget),
# gsm8k at 2048, single seed 42, n=1. At s80 truncation dominates, so comparing
# a DPO checkpoint measured at 16384 against that 8192 baseline measures the
# budget, not DPO. All three arms here run the identical protocol.
#
# Usage: bash eval_s80_long.sh <arm> [CUDA_VISIBLE_DEVICES]
#   arm = base | dpo_lr1e5_ep04 | dpo_lr5e6_ep10
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"

ARM=${1:?"usage: eval_s80_long.sh <base|dpo_lr1e5_ep04|dpo_lr5e6_ep10> [GPU_IDS]"}
[ $# -ge 2 ] && export CUDA_VISIBLE_DEVICES=$2

case "$ARM" in
  base)
    MODEL=cosmos1030/gmp-kd3e-1-s80pct-lr1e-4_20260916_220740
    RUN=s3_4b_s80_ours ;;
  dpo_lr1e5_ep04)
    MODEL=Riasok/dpo-scout-s80-ultrafeedback-lr1e-5-beta0.05-epoch0.4_20260918
    RUN=s3_4b_s80_ours_dpo_lr1e5_ep04 ;;
  dpo_lr5e6_ep10)
    MODEL=Riasok/dpo-scout-s80-ultrafeedback-lr5e-6-beta0.05-epoch1.0_20260918
    RUN=s3_4b_s80_ours_dpo_lr5e6_ep10 ;;
  *) die "unknown arm '$ARM' (base | dpo_lr1e5_ep04 | dpo_lr5e6_ep10)" ;;
esac

LOG="$OUT_ROOT/${RUN}_$(date +%Y%m%d_%H%M%S).log"
echo "=== s80 long eval ==="
echo "  arm      $ARM"
echo "  model    $MODEL"
echo "  run      $RUN  ->  $WANDB_PROJECT"
echo "  profile  $PROFILE   seeds $SEEDS   tp $TP_SIZE   gpu_util $GPU_UTIL"
echo "  gpus     ${CUDA_VISIBLE_DEVICES:-all}"
echo "  log      $LOG"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true

cd "$REPO_ROOT/elsa"
# PPL and zero-shot are skipped: the s3_* table is reasoning-only, and the
# zero-shot suite needs datasets this container may not have cached.
# Benchmarks are left at the default so the 'long' table decides the set
# (math500/gpqa/ifeval at 16384, lcb at 32768, gsm8k at 8192).
"$PYTHON" scripts/eval_full.py \
    --model_path "$MODEL" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_entity "$WANDB_ENTITY" \
    --run_name "$RUN" \
    --method gmp --sparsity 0.8 \
    --profile "$PROFILE" \
    --seeds "$SEEDS" \
    --tp_size "$TP_SIZE" --gpu_util "$GPU_UTIL" \
    --skip_ppl --skip_zeroshot \
    --out_base "$OUT_ROOT/eval_${RUN}" 2>&1 | tee "$LOG"

CODE=${PIPESTATUS[0]}
echo "=== EXIT: $CODE ===  log: $LOG"
exit $CODE
