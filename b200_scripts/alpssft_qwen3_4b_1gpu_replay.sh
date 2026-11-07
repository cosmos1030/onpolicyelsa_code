#!/bin/bash
# Qwen3-4B ALPS -> NTP+KD+OPD retraining (ALPS+retrain baseline) on 1 GPU,
# replaying the exact args of the 2026-10-07 roll2048 reference run
# (wandb reasoning_qwen3_4b_nostrip8192: s70 ma7dalx2) with only paths swapped to
# b200. Extra main.py flags (e.g. --gmp_ntp_slm_ratio=0.6) are appended from "$@".
#
# Usage: CUDA_VISIBLE_DEVICES=<gpu> CKPT_DIR=<dir> HUB_ID=<repo> TAG=<suffix> \
#   bash b200_scripts/alpssft_qwen3_4b_1gpu_replay.sh <SPARSITY> <LR> [extra main.py flags...]
set -o pipefail
SPARSITY=${1:?sparsity}; LR=${2:?lr}; shift 2
: "${CKPT_DIR:?CKPT_DIR required}" "${HUB_ID:?HUB_ID required}" "${TAG:?TAG required}"
SPARSITY_PCT=$(python3 -c "print(int(round(${SPARSITY}*100)))")
ALPS_MODEL=${ALPS_MODEL:-/NHNHOME/log-postech/doyoonkim/models/qwen3_4b_alps_s${SPARSITY_PCT}pct}

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
PYTHON=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

JOB_TAG="alpssft_4b_b200_s${SPARSITY_PCT}_lr${LR}_${TAG}"
LOCAL_JOB_BASE="/NHNHOME/log-postech/doyoonkim/logs/${JOB_TAG}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)
export WANDB_API_KEY=$(cat /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key)
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/torchinductor
export VLLM_CACHE_ROOT=/NHNHOME/log-postech/doyoonkim/.cache/vllm
export HF_HOME=/NHNHOME/log-postech/doyoonkim/.cache/huggingface
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export VLLM_NO_USAGE_STATS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NLTK_DATA=/NHNHOME/log-postech/doyoonkim/nltk_data

if $PYTHON -c "import sys;from huggingface_hub import HfApi;HfApi().model_info(sys.argv[1])" "$HUB_ID" >/dev/null 2>&1; then
    HUB_ID="${HUB_ID}-$(date +%m%d%H%M)"
fi
echo "=== Qwen3-4B ALPS+retrain replay s${SPARSITY_PCT} lr=${LR} tag=${TAG} init=${ALPS_MODEL} extra=[$*] GPU=${CUDA_VISIBLE_DEVICES} ==="
echo "HUB_ID=$HUB_ID"

cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/elsa
$PYTHON -u main.py \
    --model=${ALPS_MODEL} \
    --gmp_teacher_model=Qwen/Qwen3-4B \
    --dataset=mixed_cot \
    --data_path=/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --seqlen=8192 \
    --gmp_gradient_checkpointing=true \
    --gmp_opkd_vllm_sidecar=true \
    --gmp_ckpt_every_steps=256 \
    --gmp_ckpt_dir=${CKPT_DIR} \
    --gmp_resume_from=${RESUME_FROM:-} \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_max_new_tokens=2048 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path=/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl \
    --gmp_use_fsdp=false \
    --gmp_save_path=/NHNHOME/log-postech/doyoonkim/models \
    --save_model=true \
    --push_to_hub=true \
    --hub_model_id="$HUB_ID" \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b_nostrip8192 \
    --run_name_suffix="alpssft_s${SPARSITY_PCT}pct_lr${LR}_${TAG}_b200" \
    --seed=42 \
    "$@"
EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
