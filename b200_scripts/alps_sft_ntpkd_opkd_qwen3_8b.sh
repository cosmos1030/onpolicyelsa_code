#!/bin/bash
# Local (non-SLURM) single-GPU adaptation of
# elsa/scripts/slurm_alps_sft_ntpkd_opkd_qwen3_8b_fsdp2gpu.sh for a single
# B200 in this docker container: unstructured-sparsity ALPS -> SFT
# NTP+KD+OPKD(0.33/0.33/0.33) recovery, parameterized by SPARSITY (matches
# the s50/s60/s70 checkpoints from the original TR-GMP-comparison plan, see
# runs_db_qwen3_8b_nostrip8192_b200.json). That log_cluster script uses
# torchrun --nproc_per_node=2 (2xH200 FSDP, vLLM sidecar) because single-GPU
# 8B + full on-policy KD OOM'd at ~136-141GB peak on that cluster's
# hardware -- this container's single B200 has 183GB, more than that peak,
# so this runs through main.py's plain single-GPU path (--gmp_use_fsdp=false,
# vLLM built in-process) instead. Mirrors
# elsa/scripts/slurm_alps_sft_ntpkd_opkd_qwen3_4b.sh's single-GPU recipe,
# scaled to 8B. See b200_scripts/README.md "Single-GPU vs the 2-GPU FSDP
# scripts this folder mirrors" before changing GPU-count-related flags.
# Machine-local launcher (paths under /NHNHOME/log-postech/doyoonkim/).
#
# Usage: bash b200_scripts/alps_sft_ntpkd_opkd_qwen3_8b.sh <SPARSITY> [LR] [OPD_GEN_LEN] \
#          [LR_SCHEDULER] [DATA_PATH] [SEQLEN] [MASK_INTERVAL] [WANDB_PROJECT]
# e.g.: bash b200_scripts/alps_sft_ntpkd_opkd_qwen3_8b.sh 0.6 5e-5
set -e

SPARSITY=${1:?"Usage: alps_sft_ntpkd_opkd_qwen3_8b.sh <SPARSITY> [LR] [OPD_GEN_LEN] [LR_SCHEDULER] [DATA_PATH] [SEQLEN] [MASK_INTERVAL] [WANDB_PROJECT]"}
LR=${2:-1e-4}
OPD_GEN_LEN=${3:-256}
LR_SCHEDULER=${4:-cosine}
DATA_PATH=${5:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${6:-8192}
MASK_INTERVAL=${7:-32}
WANDB_PROJECT=${8:-reasoning_qwen3_8b_nostrip8192}

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
PYTHON=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

SPARSITY_PCT=$($PYTHON -c "print(int(${SPARSITY}*100))")
ALPS_MODEL="/NHNHOME/log-postech/doyoonkim/models/qwen3_8b_alps_s${SPARSITY_PCT}pct"
SPARSITY_TAG="s${SPARSITY_PCT}pct"

DENSE_MODEL=$(ls -d /NHNHOME/log-postech/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ 2>/dev/null | head -1)
DENSE_MODEL="${DENSE_MODEL%/}"
if [ -z "$DENSE_MODEL" ] || [ ! -f "$DENSE_MODEL/config.json" ]; then
    echo "ERROR: Qwen3-8B not found in HF cache" >&2
    exit 1
fi
if [ ! -d "$ALPS_MODEL" ]; then
    echo "ERROR: ALPS checkpoint not found at $ALPS_MODEL" >&2
    exit 1
fi

OPD_PROMPT_PATH="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

JOB_TAG="alpssft_8b_b200_${SPARSITY_TAG}_lr${LR}"
LOCAL_JOB_BASE="/NHNHOME/log-postech/doyoonkim/logs/${JOB_TAG}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)
export WANDB_API_KEY=$(cat /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key)
# NOTE: expandable_segments left UNSET -- vLLM's CuMemAllocator
# (enable_sleep_mode=True), which the single-GPU OPKD path uses, hard-asserts
# against it at load_model() time. See b200_scripts/README.md and
# slurm_alps_sft_ntpkd_opkd_qwen3_4b.sh.
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/torchinductor
export VLLM_CACHE_ROOT=/NHNHOME/log-postech/doyoonkim/.cache/vllm
export HF_HOME="/NHNHOME/log-postech/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

echo "=== ALPS -> Sparse SFT NTP+KD+OPKD(0.33/0.33/0.33) Qwen3-8B ${SPARSITY_TAG} lr=${LR} opd_gen_len=${OPD_GEN_LEN} seqlen=${SEQLEN} -- 1xB200 single-GPU, vLLM in-process ==="
echo "NODE=$(hostname)  MODEL=$ALPS_MODEL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/elsa

$PYTHON main.py \
    --model="$ALPS_MODEL" \
    --gmp_teacher_model="$DENSE_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --gmp_use_fsdp=false \
    --steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_kd_only=false \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_save_path=/NHNHOME/log-postech/doyoonkim/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_profile=quick \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --seed=42 \
    --run_name_suffix="alpssft_${SPARSITY_TAG}_lr${LR}_$(basename "$DATA_PATH" .jsonl)_b200"

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
