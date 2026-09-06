#!/bin/bash
# Re-run the post-pruning lighteval bench + HF upload for a model whose training
# already finished and was saved, but whose eval was cut short (container死 /
# OOM / manual kill). No retraining: reads the saved model dir off disk and logs
# into the same wandb run the training used.
#
# Env below mirrors gmp_pgd_grow_to_target_qwen3_8b_fsdp4gpu.sh, minus the
# torchrun distributed vars -- vLLM brings up its own process group and chokes on
# inherited RANK/WORLD_SIZE (see lib/lighteval_bench.py's _run_lighteval).
#
# Usage: bash b200_scripts/resume_eval_lighteval.sh <MODEL_DIR> <WANDB_RUN_ID> [extra args to the .py]
# e.g.  bash b200_scripts/resume_eval_lighteval.sh \
#         /NHNHOME/log-postech/doyoonkim/models/gmp_s70pct_..._20260906_105649 e9xuky3e \
#         --sparsity 0.7 --lr 1e-4
set -e

MODEL_DIR=${1:?"Usage: <MODEL_DIR> <WANDB_RUN_ID> [extra args]"}
WANDB_RUN_ID=${2:?"Usage: <MODEL_DIR> <WANDB_RUN_ID> [extra args]"}
shift 2

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
PY=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)
export WANDB_API_KEY=$(cat /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/torchinductor
export VLLM_CACHE_ROOT=/NHNHOME/log-postech/doyoonkim/.cache/vllm
export HF_HOME=/NHNHOME/log-postech/doyoonkim/.cache/huggingface
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export VLLM_NO_USAGE_STATS=1
export NCCL_DEBUG=WARN
# lighteval pulls task datasets from the hub
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

echo "=== resume eval: $MODEL_DIR (wandb run $WANDB_RUN_ID) ==="
echo "NODE=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/elsa
exec $PY /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/b200_scripts/resume_eval_lighteval.py \
    --model_dir "$MODEL_DIR" --wandb_run_id "$WANDB_RUN_ID" "$@"
