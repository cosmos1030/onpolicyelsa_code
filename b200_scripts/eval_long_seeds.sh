#!/bin/bash
# Eval-only (long profile, reasoning benchmarks) of one checkpoint at extra seeds (default 0,1) -- adds to the
# seed-42 in-training eval so every roll2048 row can be reported as a 3-seed mean (42/0/1).
# Usage: CUDA_VISIBLE_DEVICES=<gpu> [SEEDS=0,1] [GPU_UTIL=0.90] bash eval_long_seeds.sh <model> <run_name> <method> <sparsity> <qwen|olmo>
# CUDA graphs on (lighteval hard-codes enforce_eager=True): same sampling/logic, ~25-30% more tokens/s in real evals.
# If the graph run fails, it is retried once in eager mode.
set -u
R=/NHNHOME/log-postech/doyoonkim
MODEL=$1; RUN=$2; METHOD=$3; SP=$4; FAM=$5; SEEDS=${SEEDS:-0,1}; GPU_UTIL=${GPU_UTIL:-0.90}
case $FAM in qwen) ENV=rac; PROJ=reasoning_qwen3_8b_nostrip8192;; olmo) ENV=rac_olmo3; PROJ=reasoning_olmo3_7b_nostrip8192;; r1llama) ENV=rac; PROJ=reasoning_r1llama_8b_nostrip8192;; *) echo "bad family $FAM"; exit 2;; esac
source $R/miniconda3/etc/profile.d/conda.sh; conda activate $ENV
export HF_TOKEN=$(cat $R/secrets/hf_token) WANDB_API_KEY=$(cat $R/secrets/wandb_api_key)
export HF_HOME=$R/.cache/huggingface VLLM_CACHE_ROOT=$R/.cache/vllm TRITON_CACHE_DIR=$R/.cache/triton TORCHINDUCTOR_CACHE_DIR=$R/.cache/torchinductor
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # make CUDA_VISIBLE_DEVICES match nvidia-smi indices
export NLTK_DATA=$R/nltk_data TMPDIR=/tmp TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_HOST_IP=127.0.0.1
cd $R/onpolicyelsa_code/elsa
run(){ python scripts/eval_full.py --model_path "$MODEL" --wandb_project $PROJ --run_name "$RUN" \
  --method "$METHOD" --sparsity "$SP" --profile long --seeds "$SEEDS" --tp_size 1 --gpu_util "$GPU_UTIL" \
  ${BENCH:+--benchmarks $BENCH} --skip_ppl --skip_zeroshot --out_base $R/logs/eval_seeds01/$RUN; }
nres(){ for s in ${SEEDS//,/ }; do for b in ${BENCH:-math500,gpqa,ifeval,lcb,gsm8k}; do :; done; done; find $R/logs/eval_seeds01/$RUN -name "results_*.json" -newer $R/logs/eval_seeds01/.start_$RUN 2>/dev/null | wc -l; }
touch $R/logs/eval_seeds01/.start_$RUN
NB=$( [ -n "${BENCH:-}" ] && echo "$BENCH" | tr , "\n" | wc -l || echo 5 )
NEED=$(( $(echo "$SEEDS" | tr , "\n" | wc -l) * NB ))
ELSA_VLLM_CUDA_GRAPH=1 run; RC=$?
# eval_full.py exits 0 even when every benchmark failed, so judge by the result files
if [ $RC -ne 0 ] || [ "$(nres)" -lt "$NEED" ]; then echo "=== graph run failed (rc=$RC) -- retrying eager ==="; ELSA_VLLM_CUDA_GRAPH=0 run; RC=$?; fi
[ "$(nres)" -lt "$NEED" ] && RC=1
echo "=== EXIT: $RC ==="
