#!/bin/bash
# Rerun of just the ifeval lighteval task for the s60pct checkpoint --
# the first eval_full.py pass failed this one task (missing `langdetect`
# package, now installed) while the rest succeeded. See eval_gmp_s60pct.sh.
set -e
source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
PYTHON=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)
export WANDB_API_KEY=$(cat /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key)
export HF_HOME=/NHNHOME/log-postech/doyoonkim/.cache/huggingface
export VLLM_CACHE_ROOT=/NHNHOME/log-postech/doyoonkim/.cache/vllm
export TRITON_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/torchinductor
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/elsa

$PYTHON scripts/eval_full.py \
    --model_path /NHNHOME/log-postech/doyoonkim/models/gmp_s60pct_lr5e-05_onpol_lmda0.33_20260820_033814 \
    --wandb_project reasoning_qwen3_8b_nostrip8192 \
    --wandb_run_id 1v1lgjqt \
    --method gmp --sparsity 0.6 \
    --tp_size 1 --gpu_util 0.85 \
    --profile quick \
    --skip_ppl --skip_zeroshot \
    --benchmarks ifeval \
    --out_base /NHNHOME/log-postech/doyoonkim/logs/alpssft_8b_b200_s60pct_lr5e-5/eval_out \
    --hub_model_id cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260820_033814 \
    --hub_url https://huggingface.co/cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260820_033814

EXIT_CODE=$?
echo "=== eval_full.py (ifeval only) EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
