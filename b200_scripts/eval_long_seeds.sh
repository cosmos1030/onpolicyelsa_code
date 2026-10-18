#!/bin/bash
# Eval-only (long profile, reasoning benchmarks) of one checkpoint at extra seeds (default 0,1) -- adds to the
# seed-42 in-training eval so every roll2048 row can be reported as a 3-seed mean (42/0/1).
# Usage: CUDA_VISIBLE_DEVICES=<gpu> [SEEDS=0,1] bash eval_long_seeds.sh <model> <run_name> <method> <sparsity> <qwen|olmo>
set -u
R=/NHNHOME/log-postech/doyoonkim
MODEL=$1; RUN=$2; METHOD=$3; SP=$4; FAM=$5; SEEDS=${SEEDS:-0,1}
case $FAM in qwen) ENV=rac; PROJ=reasoning_qwen3_8b_nostrip8192;; olmo) ENV=rac_olmo3; PROJ=reasoning_olmo3_7b_nostrip8192;; *) echo "bad family $FAM"; exit 2;; esac
source $R/miniconda3/etc/profile.d/conda.sh; conda activate $ENV
export HF_TOKEN=$(cat $R/secrets/hf_token) WANDB_API_KEY=$(cat $R/secrets/wandb_api_key)
export HF_HOME=$R/.cache/huggingface VLLM_CACHE_ROOT=$R/.cache/vllm TRITON_CACHE_DIR=$R/.cache/triton TORCHINDUCTOR_CACHE_DIR=$R/.cache/torchinductor
# CUDA graphs on (lighteval hard-codes enforce_eager=True); same sampling/logic, ~17-20% faster generation (measured 2026-10-18)
[ "$FAM" = qwen ] && export ELSA_VLLM_CUDA_GRAPH=1   # OLMo (V0 plugin) not yet verified under CUDA graphs
export NLTK_DATA=$R/nltk_data TMPDIR=/tmp TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_HOST_IP=127.0.0.1
cd $R/onpolicyelsa_code/elsa
python scripts/eval_full.py --model_path "$MODEL" --wandb_project $PROJ --run_name "$RUN" \
  --method "$METHOD" --sparsity "$SP" --profile long --seeds "$SEEDS" --tp_size 1 --gpu_util 0.90 \
  --skip_ppl --skip_zeroshot --out_base $R/logs/eval_seeds01/$RUN
echo "=== EXIT: $? ==="
