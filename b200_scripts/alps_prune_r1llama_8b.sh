#!/bin/bash
# DeepSeek-R1-Distill-Llama-8B fork of alps_prune_r1llama_8b.sh (2026-10-20): same recipe and positional args;
# env rac (vLLM 0.10 / transformers 4.55 support Llama natively), R1-format data
# (ot3_fineweb_40k_r1llama_nostrip_8192 = same OT3/FineWeb rows, Llama tokenizer; opd prompts = same OT3 prompts
# in <｜User｜>...<｜Assistant｜><think>\n form), wandb project reasoning_r1llama_8b_nostrip8192.
# OLMo 3 7B Think fork of alps_prune_qwen3_8b.sh: one-shot ALPS pruning, then
# save (the starting point for alps_sft_ntpkd_opkd_r1llama_8b_fsdp2gpu.sh).
# Env rac (transformers 4.57 + vllm_olmo3_plugin). ALPS/qwen3_alps.py
# replays layer kwargs per attention type (OLMo 3 mixes sliding/full layers
# with different rope). The one-shot eval is OFF by default (ALPS_EVAL=1 turns
# it on) so the SFT stage can start as soon as the model is saved.
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
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DATA="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_r1llama_nostrip_8192.jsonl"
SAVE_BASE="/NHNHOME/log-postech/doyoonkim/models"
SAVED_MODEL=${SAVED_MODEL:-"${SAVE_BASE}/r1llama_8b_alps_s${SPARSITY_PCT}pct"}

JOB_TAG="alps_r1llama_8b_b200_${SPARSITY_PCT}pct"
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
# This host runs heavily multi-tenant (other users' jobs contending for the
# same physical cores) -- default thread pools sized to nproc (72 here) blow
# up badly under that contention (measured: single-call tokenizer/model-load
# ops that take seconds in isolation took 1-2+ HOURS here with 0% GPU util
# the whole time). Caps every native thread pool we know of, not just torch's.
export OMP_NUM_THREADS=${CPU_THREADS:-4}
export MKL_NUM_THREADS=${CPU_THREADS:-4}
export OPENBLAS_NUM_THREADS=${CPU_THREADS:-4}
export NUMEXPR_NUM_THREADS=4
export RAYON_NUM_THREADS=4
export TRITON_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/torchinductor
export VLLM_CACHE_ROOT=/NHNHOME/log-postech/doyoonkim/.cache/vllm
export HF_HOME="/NHNHOME/log-postech/doyoonkim/.cache/huggingface"
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
# ALPS/qwen3_alps.py reads this env var (falls back to another server's path
# if unset) -- see b200_scripts/README.md.
export EVAL_FULL_SCRIPT=/NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/elsa/scripts/eval_full.py

echo "=== ALPS Pruning R1-Distill-Llama-8B s${SPARSITY_PCT}% (1xB200, local container run) ==="
echo "NODE=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/ALPS

HUB_ID=${HUB_ID:-cosmos1030/r1llama-8b-alps-oneshot-s${SPARSITY_PCT}}
# never overwrite an existing repo: append a timestamp if the name is taken
if $PYTHON -c "import sys;from huggingface_hub import HfApi;HfApi().model_info(sys.argv[1])" "$HUB_ID" >/dev/null 2>&1; then
    HUB_ID="${HUB_ID}-$(date +%m%d%H%M)"
fi
echo "HUB_ID=$HUB_ID"
$PYTHON -u qwen3_alps.py \
    "$MODEL" \
    ${SPARSITY} \
    --data_path "$DATA" \
    --nsamples 128 \
    --rho 300.0 \
    --seed 42 \
    --save "$SAVED_MODEL" \
    $([ "${ALPS_EVAL:-0}" = "1" ] && echo --eval_full) \
    --wandb_project reasoning_r1llama_8b_nostrip8192 \
    --run_name "alps_r1llama_8b_s${SPARSITY_PCT}pct_b200" \
    --gpu_util 0.85 \
    --tp_size 1 \
    --out_base "$LOCAL_JOB_BASE/eval_out" \
    --profile ${EVAL_PROFILE:-long} \
    --push_to_hub \
    --hub_model_id "$HUB_ID"

EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
