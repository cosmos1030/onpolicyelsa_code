#!/bin/bash
# Standalone eval (zero-shot + lighteval quick + ppl) for the s60pct SFT
# checkpoint whose in-training eval crashed (TRANSFORMERS_OFFLINE blocked
# allenai/ai2_arc, which isn't cached in this container -- see
# b200_scripts/README.md). Resumes the same wandb run so results land
# alongside the training curve.
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
# Deliberately NOT setting HF_DATASETS_OFFLINE/TRANSFORMERS_OFFLINE -- this
# container has internet and needs it to fetch eval datasets not already cached.

cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/elsa

$PYTHON scripts/eval_full.py \
    --model_path /NHNHOME/log-postech/doyoonkim/models/gmp_s60pct_lr5e-05_onpol_lmda0.33_20260820_033814 \
    --wandb_project reasoning_qwen3_8b_nostrip8192 \
    --wandb_run_id 1v1lgjqt \
    --method gmp --sparsity 0.6 \
    --tp_size 1 --gpu_util 0.85 \
    --profile quick \
    --out_base /NHNHOME/log-postech/doyoonkim/logs/alpssft_8b_b200_s60pct_lr5e-5/eval_out \
    --hub_model_id cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260820_033814 \
    --hub_url https://huggingface.co/cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260820_033814

EXIT_CODE=$?
echo "=== eval_full.py EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
