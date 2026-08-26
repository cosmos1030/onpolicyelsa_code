#!/bin/bash
# Local (non-SLURM) single-GPU adaptation of
# elsa/scripts/log_cluster/slurm_gmp_tr_ntpkd_opd_qwen3_8b_fsdp2gpu.sh for a
# single B200 in this docker container: TR-GMP (trust-region gradual mask
# growth, starts from DENSE -- not ALPS-initialized) NTP+KD+OPD
# (0.33/0.33/0.33), 2:4 semi-structured. Args mirror the finished 4B N:M
# reference run (wandb run fb3peyhp, reasoning_qwen3_4b_nostrip8192),
# scaled to 8B -- notably --gmp_l1_lambda=0.0001 (structured L1 toward 2:4
# groups, off by default, only that 4B N:M run enabled it; unstructured 8B
# TR-GMP scripts in this repo don't set it).
#
# That log_cluster script uses torchrun --nproc_per_node=2 (2xH200 FSDP,
# vLLM sidecar) because a single-H200 attempt (jobs 41450-41455) OOM'd at
# step 1 inside the full-vocab KD KL loss (~136GB peak, seqlen=8192 x
# ~152k vocab). This container's single B200 has 183GB, more than that
# peak, so this runs through main.py's plain single-GPU path
# (--gmp_use_fsdp=false, vLLM built in-process) instead -- same reasoning
# already validated for the ALPS->SFT 8B scripts in this folder. See
# b200_scripts/README.md before changing GPU-count-related flags.
# Machine-local launcher (paths under /NHNHOME/log-postech/doyoonkim/).
#
# Usage: bash b200_scripts/tr_gmp_ntpkd_opd_qwen3_8b_24.sh [LR] [KL_THRESHOLD] [MASK_INTERVAL] \
#          [LR_SCHEDULER] [SALIENCY] [DATA_PATH] [SEQLEN] [WANDB_PROJECT]
# e.g.: bash b200_scripts/tr_gmp_ntpkd_opd_qwen3_8b_24.sh 1e-4 0.02
set -e

LR=${1:-1e-4}
KL_THRESHOLD=${2:-0.02}
MASK_INTERVAL=${3:-32}
LR_SCHEDULER=${4:-cosine}
SALIENCY=${5:-fisher}
DATA_PATH=${6:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${7:-8192}
WANDB_PROJECT=${8:-reasoning_qwen3_8b_nostrip8192}

SPARSITY=0.5
SPARSITY_TYPE=2:4
MODEL="Qwen/Qwen3-8B"

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
PYTHON=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

OPD_PROMPT_PATH="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

JOB_TAG="trgmp_8b_b200_n24_lr${LR}_kl${KL_THRESHOLD}"
LOCAL_JOB_BASE="/NHNHOME/log-postech/doyoonkim/logs/${JOB_TAG}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)
export WANDB_API_KEY=$(cat /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key)
# NOTE: expandable_segments left UNSET -- vLLM's CuMemAllocator
# (enable_sleep_mode=True), which the single-GPU OPD path uses, hard-asserts
# against it at load_model() time. See b200_scripts/README.md.
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/torchinductor
export VLLM_CACHE_ROOT=/NHNHOME/log-postech/doyoonkim/.cache/vllm
export HF_HOME=/NHNHOME/log-postech/doyoonkim/.cache/huggingface
# Deliberately NOT setting HF_DATASETS_OFFLINE/TRANSFORMERS_OFFLINE -- this
# container has internet and needs it to fetch eval datasets not already
# cached (bit us on the s60pct ALPS->SFT run's zero-shot eval, see
# eval_gmp_s60pct.sh).
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

echo "=== TR-GMP NTP+KD+OPD (0.33/0.33/0.33) Qwen3-8B 2:4 semi-structured lr=${LR} kl=${KL_THRESHOLD} mi=${MASK_INTERVAL} saliency=${SALIENCY} -- 1xB200 single-GPU, vLLM in-process ==="
echo "NODE=$(hostname)  MODEL=$MODEL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=${SPARSITY_TYPE} \
    --gmp_l1_lambda=0.0001 \
    --do_gmp=true \
    --gmp_use_fsdp=false \
    --steps=2048 \
    --gmp_post_target_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_kd_only=false \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_saliency=${SALIENCY} \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
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
    --run_name_suffix="${SALIENCY}_n24_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}_b200"

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
