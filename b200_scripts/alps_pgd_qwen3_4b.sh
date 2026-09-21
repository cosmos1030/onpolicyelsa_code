#!/bin/bash
# ALPS mask + trust-region PGD -- isolates what PGD contributes with the mask
# SOURCE held fixed.
#
# ALPS+retrain already runs the same NTP+KD+OPD recovery on a frozen ALPS mask
# (44.94 avg5 at 4B s70, long/3 seeds) and SCOUT reaches 48.81 by growing its
# own mask from 0% with KL-gated PGD. The two differ in both the mask and the
# projection, so neither number attributes the gap. This arm keeps the ALPS
# mask as the starting point and turns PGD on, which splits that gap in two:
# ALPS+retrain -> here is PGD at a fixed mask origin, here -> SCOUT is what
# growing from 0% adds.
#
# gmp_fixed_mask=true only seeds the mask from the checkpoint's zeros
# (maskmgr.init_from_weights) and disables the SCHEDULE-driven growth; it does
# not gate PGD, whose run condition reads pgd_enabled and the interval alone.
# grow_to_target=true is correct even though the model already sits at target:
# per the flag's own docs, prune_cand/revive_cand converge there and it
# "degrades to pure polish/maintenance -- no separate at-target branch needed",
# i.e. the same code path SCOUT runs after step 224.
#
# Two arms from one script:
#   KL_BUDGET=0.02  (default)  trust region on, matches SCOUT s70
#   KL_BUDGET=99999            trust region removed; bisection accepts every
#                              candidate, so the projection is ungated
#
# Checkpoints at 512/1024/1536 via --gmp_milestone_steps (override MILESTONE_STEPS).
#
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
# 512, not 256: every 8B PGD run in this project passed 512 explicitly, and the
# 4B launchers already default to 512 -- leaving the default at 256 meant the
# ALPS+SFT baselines silently trained on HALF the on-policy KD tokens per
# rollout that the method they are compared against used (same 256 rollouts per
# refill window either way, but 256 vs 512 tokens each). Do not lower it back
# without re-running both sides.
OPD_GEN_LEN=${3:-512}
LR_SCHEDULER=${4:-cosine}
DATA_PATH=${5:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${6:-8192}
MASK_INTERVAL=${7:-32}
WANDB_PROJECT=${8:-reasoning_qwen3_8b_nostrip8192}

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
PYTHON=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

SPARSITY_PCT=$($PYTHON -c "print(int(${SPARSITY}*100))")
ALPS_MODEL="/NHNHOME/log-postech/doyoonkim/models/qwen3_4b_alps_s${SPARSITY_PCT}pct"
SPARSITY_TAG="s${SPARSITY_PCT}pct"
# --- this fork's own knobs -------------------------------------------------
# KL_BUDGET: the trust region on each PGD projection. 0.02 matches SCOUT s70.
#   Pass 99999 for the no-trust-region arm: the bisection still runs (same code
#   path) but the constraint never binds, so every candidate swap is accepted.
KL_BUDGET="${KL_BUDGET:-0.02}"
PGD_INTERVAL="${PGD_INTERVAL:-8}"
CALIB_SIZE="${CALIB_SIZE:-4}"
MILESTONE_STEPS="${MILESTONE_STEPS:-512,1024,1536}"   # 원본의 --gmp_milestone_steps 가 읽는다
ARM_TAG="${ARM_TAG:-pgd_klb${KL_BUDGET}}"

DENSE_MODEL=$(ls -d /NHNHOME/log-postech/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/*/ 2>/dev/null | head -1)
DENSE_MODEL="${DENSE_MODEL%/}"
if [ -z "$DENSE_MODEL" ] || [ ! -f "$DENSE_MODEL/config.json" ]; then
    echo "ERROR: Qwen3-4B not found in HF cache" >&2
    exit 1
fi
if [ ! -d "$ALPS_MODEL" ]; then
    echo "ERROR: ALPS checkpoint not found at $ALPS_MODEL" >&2
    exit 1
fi

OPD_PROMPT_PATH="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

JOB_TAG="alpspgd_4b_b200_${SPARSITY_TAG}_lr${LR}_${ARM_TAG}${TAG_SUFFIX:-}"
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

echo "=== ALPS -> Sparse SFT NTP+KD+OPKD(${NTP_LAMBDA:-0.33}/${KD_LAMBDA:-0.33}/${OPKD_LAMBDA:-0.33}) Qwen3-4B ${SPARSITY_TAG} lr=${LR} opd_gen_len=${OPD_GEN_LEN} seqlen=${SEQLEN} -- 1xB200 single-GPU, vLLM in-process ==="
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
    --gmp_tr_enabled=false \
    --gmp_pgd=true \
    --gmp_pgd_grow_to_target=true \
    --gmp_pgd_kl_budget=${KL_BUDGET} \
    --gmp_pgd_interval=${PGD_INTERVAL} \
    --gmp_pgd_kl_calib_size=${CALIB_SIZE} \
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
    --gmp_ntp_lambda=${NTP_LAMBDA:-0.33} \
    --gmp_kd_lambda=${KD_LAMBDA:-0.33} \
    --gmp_onpolicy_kd_lambda=${OPKD_LAMBDA:-0.33} \
    --gmp_milestone_steps="${MILESTONE_STEPS:-}" \
    --gmp_onpolicy_kd_interval=${ROLLOUT_INTERVAL:-32} \
    --gmp_kd_only=false \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_save_path=/NHNHOME/log-postech/doyoonkim/models \
    --gmp_ckpt_every_steps=${CKPT_EVERY:-0} --gmp_ckpt_dir="${CKPT_DIR:-}" --gmp_resume_from="${RESUME_FROM:-}" \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_profile=${EVAL_PROFILE:-long} \
    --eval_zero_shot=${EVAL_ZERO_SHOT:-false} \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --seed=42 \
    --run_name_suffix="alpspgd4b_${SPARSITY_TAG}_lr${LR}_${ARM_TAG}${TAG_SUFFIX:-}_$(basename "$DATA_PATH" .jsonl)_b200"

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
