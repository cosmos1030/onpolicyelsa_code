#!/bin/bash
# Pure A3jump (KL-gate jump rule, no frozen pool) 8B ablation arms, long profile.
#
# The RESULTS_ALL.md numbers for these are quick-profile (8192 / gsm8k 2048),
# so they cannot be put next to the s3_8b_* table, which is long. Same protocol
# as eval_8b_long.sh, different checkpoints.
#
# There is no s50 A3jump arm: queue2.sh/queue2b.sh only ever defined s70 jump,
# ablation_24 added s60 and 2:4. Do not go looking for one.
#
# Usage: bash eval_8b_a3jump.sh <s60|s70>
set -u

ROOT=${ELSA_ROOT:-/NHNHOME/log-postech/doyoonkim}
REPO=${ELSA_REPO:-$ROOT/onpolicyelsa_code}
PYTHON=${PYTHON:-$ROOT/miniconda3/envs/rac/bin/python}
OUT_ROOT=${OUT_ROOT:-$ROOT/logs/eval_8b_long}
SEEDS=${SEEDS:-42}
PROFILE=${PROFILE:-long}
GPU_UTIL=${GPU_UTIL:-0.90}
TP_SIZE=${TP_SIZE:-1}
WANDB_PROJECT=${WANDB_PROJECT:-reasoning_qwen3_8b_nostrip8192}

die () { echo "!! $*" >&2; exit 1; }
ARM=${1:?"usage: eval_8b_a3jump.sh <s60|s70>"}

case "$ARM" in
  # d=0.01, lr=5e-5. ablation_24/s60_8b_A3jump_d0.01_r1.log -> quick avg5 53.30
  s60) M=$ROOT/models/gmp_s60pct_lr5e-05_onpol_lmda0.33_20260914_002323_p2602547; SP=0.6 ;;
  # d=0.02, lr=1e-4. resweep2_opkdfix/s70_A3jump_delta0.02.log -> quick avg5 42.86.
  # NOT ablation_24/s70_8b_A3jump_d0.02_s2 -- that run died with an empty ckpt dir.
  s70) M=$ROOT/models/gmp_s70pct_lr0.0001_onpol_lmda0.33_20260908_085535;         SP=0.7 ;;
  *) die "unknown arm '$ARM' (s60 | s70; there is no s50 A3jump)" ;;
esac

[ -f "$M/config.json" ] || die "no config.json under $M"
[ "$(ls "$M"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ] || die "no safetensors under $M"

RUN="s3_8b_a3jump_${ARM}"
MARK="$OUT_ROOT/.done_${RUN}"
if [ -f "$MARK" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "== $ARM already done ($MARK) -- skipping. FORCE=1 to redo."; exit 0
fi

source "$ROOT/miniconda3/etc/profile.d/conda.sh"
conda activate rac
[ -x "$PYTHON" ] || die "no python at $PYTHON"
grep -q "_LONG_BENCHMARKS" "$REPO/elsa/lib/lighteval_bench.py" || die "checkout predates the long profile"

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
echo "=== 8B A3jump long eval ==="
echo "  arm $ARM   model $M"
echo "  run $RUN -> $WANDB_PROJECT   seeds $SEEDS   profile $PROFILE"
echo "  gpu ${CUDA_VISIBLE_DEVICES:-all}   log $LOG"

cd "$REPO/elsa"
"$PYTHON" scripts/eval_full.py \
    --model_path "$M" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "$RUN" \
    --method gmp --sparsity "$SP" \
    --profile "$PROFILE" \
    --seeds "$SEEDS" \
    --tp_size "$TP_SIZE" --gpu_util "$GPU_UTIL" \
    --skip_ppl --skip_zeroshot \
    --out_base "$OUT_ROOT/eval_${RUN}" 2>&1 | tee "$LOG"

CODE=${PIPESTATUS[0]}
if [ "$CODE" -eq 0 ] || grep -q "lighteval bench done\|_mean" "$LOG"; then touch "$MARK"; fi
echo "=== EXIT: $CODE ===  log: $LOG"
exit $CODE
