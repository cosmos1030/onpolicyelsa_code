#!/bin/bash
# Local (non-SLURM) adaptation of ALPS/slurm_alps_prune_8b_rtx6000ada.sh for a
# single B200 in this docker container. Not committed to git -- machine-local
# launcher only (model/data/save paths point at /NHNHOME persistent storage
# and Qwen/Qwen3-8B is pulled from the HF hub instead of the other server's
# local snapshot cache path).
# Usage: run_alps_prune_8b.sh <SPARSITY>
set -e

SPARSITY=${1:?"Usage: run_alps_prune_8b.sh <SPARSITY>"}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac

PYTHON=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="Qwen/Qwen3-8B"
DATA="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
SAVE_BASE="/NHNHOME/log-postech/doyoonkim/models"
SAVED_MODEL="${SAVE_BASE}/qwen3_8b_alps_s${SPARSITY_PCT}pct"

JOB_TAG="alps_8b_b200_${SPARSITY_PCT}pct"
LOCAL_JOB_BASE="/NHNHOME/log-postech/doyoonkim/logs/${JOB_TAG}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/eval_out"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)
export WANDB_API_KEY=$(cat /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/torchinductor
export VLLM_CACHE_ROOT=/NHNHOME/log-postech/doyoonkim/.cache/vllm
export HF_HOME="/NHNHOME/log-postech/doyoonkim/.cache/huggingface"
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
# ALPS/qwen3_alps.py reads this env var (falls back to another server's path
# if unset) -- see b200_scripts/README.md.
export EVAL_FULL_SCRIPT=/home/log_lab/onpolicyelsa_code/elsa/scripts/eval_full.py

echo "=== ALPS Pruning Qwen3-8B s${SPARSITY_PCT}% (1xB200, local container run) ==="
echo "NODE=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home/log_lab/onpolicyelsa_code/ALPS

$PYTHON qwen3_alps.py \
    "$MODEL" \
    ${SPARSITY} \
    --data_path "$DATA" \
    --nsamples 128 \
    --rho 300.0 \
    --seed 42 \
    --save "$SAVED_MODEL" \
    --eval_full \
    --wandb_project reasoning_qwen3_8b \
    --run_name "alps_8b_s${SPARSITY_PCT}pct_b200" \
    --gpu_util 0.85 \
    --tp_size 1 \
    --out_base "$LOCAL_JOB_BASE/eval_out" \
    --profile quick \
    --push_to_hub

EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
