#!/bin/bash
# The only file you edit per-server. Everything else reads from here.
#
# Each value can also be overridden from the shell without touching this file:
#   REPO_ROOT=/path/to/onpolicyelsa PYTHON=/path/to/python bash run_all_s80_long.sh
#
# Set REQUIRED values or the scripts abort with a message naming the missing one
# -- an eval that dies 40 minutes in on a bad path is the failure mode this
# exists to prevent.

# --- REQUIRED ---------------------------------------------------------------
# Repo checkout root: the directory that CONTAINS elsa/ (not elsa itself).
REPO_ROOT=${REPO_ROOT:-}
# Python that has vllm + lighteval + transformers installed.
PYTHON=${PYTHON:-}

# --- credentials ------------------------------------------------------------
# Either point these at files, or export HF_TOKEN / WANDB_API_KEY yourself.
HF_TOKEN_FILE=${HF_TOKEN_FILE:-$HOME/.hf_token}
WANDB_KEY_FILE=${WANDB_KEY_FILE:-}

# --- caches -----------------------------------------------------------------
# Leave blank to use this machine's defaults (~/.cache/...). Set them if the
# home filesystem is small or slow; the 4B checkpoints are 7.6GB each.
HF_HOME=${HF_HOME:-}
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-}
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-}
TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-}

# --- where results and logs go ---------------------------------------------
OUT_ROOT=${OUT_ROOT:-$HOME/elsa_eval_s80}

# --- eval settings ----------------------------------------------------------
# tp_size: 1 unless the card cannot hold 4B bf16 + a 33k-token KV cache.
# Raise only if you see "No available memory for the cache blocks".
TP_SIZE=${TP_SIZE:-1}
GPU_UTIL=${GPU_UTIL:-0.85}
SEEDS=${SEEDS:-0,1,42}
PROFILE=${PROFILE:-long}
WANDB_PROJECT=${WANDB_PROJECT:-reasoning_qwen3_4b_nostrip8192}
WANDB_ENTITY=${WANDB_ENTITY:-dyk6208-gwangju-institute-of-science-and-technology}

# ---------------------------------------------------------------------------
die () { echo "!! $*" >&2; exit 1; }

[ -n "$REPO_ROOT" ] || die "REPO_ROOT is unset -- edit log_scripts/env.sh"
[ -n "$PYTHON" ]    || die "PYTHON is unset -- edit log_scripts/env.sh"
[ -d "$REPO_ROOT/elsa" ] || die "no elsa/ under REPO_ROOT=$REPO_ROOT"
[ -x "$PYTHON" ] || die "PYTHON=$PYTHON is not executable"

# The 'long' profile is a recent addition (elsa/lib/lighteval_bench.py). An
# older checkout accepts --profile long at the argparse level and then dies in
# the library, so check for the table itself rather than the flag.
grep -q "_LONG_BENCHMARKS" "$REPO_ROOT/elsa/lib/lighteval_bench.py" \
  || die "this checkout has no 'long' profile -- git pull in $REPO_ROOT (needs commits dffbac7, e9b30c5)"

if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
  export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
fi
if [ -z "${WANDB_API_KEY:-}" ] && [ -n "$WANDB_KEY_FILE" ] && [ -f "$WANDB_KEY_FILE" ]; then
  export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE")
fi
[ -n "${HF_TOKEN:-}" ] || echo "   (warning) no HF_TOKEN -- private repos will 401"
# eval_full.py treats a failed wandb.init as fatal on purpose: results that
# never reach wandb have been silently lost before. Set EVAL_ALLOW_NO_WANDB=1
# only for a throwaway run.
[ -n "${WANDB_API_KEY:-}" ] || echo "   (warning) no WANDB_API_KEY -- eval_full.py will abort unless EVAL_ALLOW_NO_WANDB=1"

for v in HF_HOME VLLM_CACHE_ROOT TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR; do
  [ -n "${!v}" ] && export $v="${!v}"
done
export TOKENIZERS_PARALLELISM=false
export TMPDIR=${TMPDIR:-/tmp}
# These two are B200/NHN workarounds; harmless elsewhere but drop them if the
# local vLLM is new enough to want V1.
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export VLLM_HOST_IP=${VLLM_HOST_IP:-127.0.0.1}
mkdir -p "$OUT_ROOT"
