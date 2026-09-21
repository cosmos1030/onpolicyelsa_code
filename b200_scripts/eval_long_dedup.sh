#!/bin/bash
# One long-profile eval, but only if nobody has it yet.
#
#   bash b200_scripts/eval_long_dedup.sh <project> <run_name> <seed> <model_path> <method> <sparsity>
#
# Asks wandb_dup_check.py first (same TSV cell -- label, sparsity, seed -- or
# same checkpoint, finished 5/5 long or running with a live heartbeat) and
# exits 0 without touching a GPU if it is covered. Meant for anything that is
# not eval_8b_long.sh, e.g. a `train`-kind line in run_4b_curve.sh's queue,
# which passes the command through bash -c.
set -u
PROJ=$1; RUN=$2; SEED=$3; M=$4; ME=$5; SP=$6
R=/NHNHOME/log-postech/doyoonkim
PY=$R/miniconda3/envs/rac/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WANDB_API_KEY=${WANDB_API_KEY:-$(cat $R/secrets/wandb_api_key)}
export HF_TOKEN=${HF_TOKEN:-$(cat $R/secrets/hf_token)}
export HF_HOME=$R/.cache/huggingface VLLM_CACHE_ROOT=$R/.cache/vllm
export TRITON_CACHE_DIR=$R/.cache/triton TORCHINDUCTOR_CACHE_DIR=$R/.cache/torchinductor
export TMPDIR=/tmp TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_HOST_IP=127.0.0.1

OUT=$("$PY" "$HERE/wandb_dup_check.py" "$PROJ" "${RUN}_seed${SEED}" "$SEED" "$M" 2>/dev/null); RC=$?
echo "[dup-check] $(printf '%s\n' "$OUT" | tail -1)"
if [ $RC -eq 0 ] && [ "${FORCE:-0}" != "1" ]; then echo "[dup-check] SKIP -- covered elsewhere"; exit 0; fi
[ $RC -eq 2 ] && echo "[dup-check] could not reach wandb -- running anyway"

cd "$R/onpolicyelsa_code/elsa"
exec "$PY" scripts/eval_full.py --model_path "$M" --wandb_project "$PROJ" \
    --run_name "${RUN}_seed${SEED}" --method "$ME" --sparsity "$SP" \
    --profile long --seeds "$SEED" --tp_size 1 --gpu_util 0.90 \
    --skip_ppl --skip_zeroshot \
    --out_base "$R/logs/eval_8b_long/eval_${RUN}_seed${SEED}"
