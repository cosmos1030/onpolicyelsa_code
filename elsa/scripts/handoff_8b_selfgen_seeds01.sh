#!/bin/bash
# 8B selfgen v3 (ALPS / SparseGPT) s50·s60·s70 의 시드 0,1 평가.
# 허브에는 시드 42만 있어서 나머지 두 시드를 채운다. 런 이름의 _s01 접미사는
# harvest가 기존 시드42 블록에 합치도록 맞춘 것이므로 바꾸지 말 것.
# A100 6장 가정: tp_size=2 이므로 3개씩 두 파도로 돈다.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG="${LOG_DIR:-/tmp}/8b_selfgen_s01"
mkdir -p "$LOG"

run() {  # $1=repo $2=run_name $3=method $4=sparsity $5=gpus
  CUDA_VISIBLE_DEVICES="$5" python scripts/eval_full.py \
    --model_path "cosmos1030/$1" \
    --wandb_project reasoning_qwen3_8b_nostrip8192 \
    --wandb_run_name "$2" \
    --method "$3" \
    --sparsity "$4" \
    --benchmarks math500,ifeval,lcb,gsm8k \
    --seeds 0,1 \
    --tp_size 2 \
    --profile long \
    --skip_ppl \
    --skip_zeroshot > "$LOG/$2.log" 2>&1
  echo "[$2] exit=$?"
}

run alps-selfgenv3-qwen3-8b-s50pct s3_8b_s50_alps_selfgen_s01 alps 0.5 0,1 &
run alps-selfgenv3-qwen3-8b-s60pct s3_8b_s60_alps_selfgen_s01 alps 0.6 2,3 &
run alps-selfgenv3-qwen3-8b-s70pct s3_8b_s70_alps_selfgen_s01 alps 0.7 4,5 &
wait

run sgpt-selfgenv3-qwen3-8b-s50pct s3_8b_s50_sgpt_selfgen_s01 sparsegpt 0.5 0,1 &
run sgpt-selfgenv3-qwen3-8b-s60pct s3_8b_s60_sgpt_selfgen_s01 sparsegpt 0.6 2,3 &
run sgpt-selfgenv3-qwen3-8b-s70pct s3_8b_s70_sgpt_selfgen_s01 sparsegpt 0.7 4,5 &
wait
echo "ALL DONE"
