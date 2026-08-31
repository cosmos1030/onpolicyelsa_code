#!/bin/bash
# PGD-driven growth (no separate TR-GMP/cubic/cosine growth mechanism) for
# this B200 container -- fork of gmp_pgd_klgate_qwen3_4b.sh (this folder)
# with growth itself replaced by --gmp_pgd_grow_to_target=true: _pgd_desired
# targets final_sparsity directly (instead of matching current keep-count),
# and revive is no longer forced equal to prune (revive saturates at
# min(k, revive_cand) while prune keeps going up to k), so the self-KL-gated
# bisection alone drives net sparsity growth toward target, at whatever pace
# --gmp_pgd_kl_budget allows. Once current sparsity reaches target,
# prune_cand/revive_cand naturally converge and it degrades to pure
# polish/maintenance -- no separate at-target branch needed.
#
# --gmp_pruning_end_ratio=0.0 (with --gmp_tr_enabled=false) disables the
# schedule-driven growth path entirely (pruning_end_steps=0 -> the
# `step <= pruning_end_steps` check that gates maskmgr.update() is false
# from step 1 onward -- see gmp_trainer.py's mask_interval block, `else`
# branch) -- PGD, gated separately by its own step%pgd_interval==0 check, is
# left as the ONLY thing that can move the mask.
#
# Validated on the log_cluster A100 side (1.7B, elsa/scripts/
# slurm_gmp_pgd_grow_to_target_qwen3_1.7b.sh) before this fork: a 100-step
# smoke test grew sparsity 0->0.7 with no overshoot, self-KL bound 100% of
# growth-phase PGD calls and 0% of post-target maintenance calls; production
# 1.7B runs (S50/S60/S70, kl_budget 0.01 and 0.02) reproduced the same
# clean-convergence pattern at full seqlen=8192 scale. unstructured only for
# now -- NOT yet supported for sparsity_type=2:4/N:M (would additionally
# need the desired-mask target to respect per-group structural caps).
#
# **This container has no SLURM** -- run directly with `bash`. Machine-local
# launcher (paths under /NHNHOME/log-postech/doyoonkim/).
#
# Usage: bash b200_scripts/gmp_pgd_grow_to_target_qwen3_4b.sh <SPARSITY> <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [ROLLOUT_INTERVAL] [KD_NSAMPLES] [CALIB_SIZE] [PGD_INTERVAL] [VLLM_GPU_MEM]
# e.g. (S50, matched lr to the existing 4B S50 capped entry 3w6q8hdw):
#   bash b200_scripts/gmp_pgd_grow_to_target_qwen3_4b.sh 0.5 0.02 512 32 cosine 2048 5e-5 "$OT3_DATA" 8192 true reasoning_qwen3_4b_nostrip8192 fisher global 0.33,0.33,0.33 32 0 4 8 0.15
set -e

SPARSITY=${1:?"Usage: <SPARSITY> <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] ..."}
KL_BUDGET=${2:?"Usage: <SPARSITY> <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] ..."}
OPD_GEN_LEN=${3:-512}
MASK_INTERVAL=${4:-32}
LR_SCHEDULER=${5:-cosine}
STEPS=${6:-2048}
LR=${7:-1e-4}
DATA_PATH_ARG=${8:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${9:-8192}
GRAD_CKPT=${10:-true}
WANDB_PROJECT=${11:-reasoning_qwen3_4b_nostrip8192}
SALIENCY=${12:-fisher}
PRUNING_SCOPE=${13:-global}
LOSS_WEIGHTS=${14:-0.33,0.33,0.33}  # NTP,KD,OPKD
ROLLOUT_INTERVAL=${15:-${MASK_INTERVAL}}  # gmp_onpolicy_kd_interval -- defaults to mask_interval
KD_NSAMPLES=${16:-0}  # 0 = full dataset (production)
CALIB_SIZE=${17:-4}   # gmp_pgd_kl_calib_size
PGD_INTERVAL=${18:-8}  # gmp_pgd_interval -- also the effective growth cadence in this mode (no separate mask_interval-triggered growth)
VLLM_GPU_MEM=${19:-0.15}  # gmp_opkd_vllm_gpu_mem -- 0.15 sized for OPD_GEN_LEN=256, raise for 512+

NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
DATA_PATH="$DATA_PATH_ARG"

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
PYTHON=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

MODEL="Qwen/Qwen3-4B"
OPD_PROMPT_PATH="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

JOB_TAG="gmp_pgd_grow_4b_b200_s${SPARSITY_PCT}_lr${LR}_klb${KL_BUDGET}_pgdi${PGD_INTERVAL}"
LOCAL_JOB_BASE="/NHNHOME/log-postech/doyoonkim/logs/${JOB_TAG}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)
export WANDB_API_KEY=$(cat /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key)
# expandable_segments left UNSET on purpose -- vLLM's CuMemAllocator
# (enable_sleep_mode=True) hard-asserts against it at load_model() time on
# the single-GPU in-process vLLM path this script uses. See README.md.
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/torchinductor
export VLLM_CACHE_ROOT=/NHNHOME/log-postech/doyoonkim/.cache/vllm
export HF_HOME=/NHNHOME/log-postech/doyoonkim/.cache/huggingface
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export VLLM_NO_USAGE_STATS=1

echo "=== PGD-driven growth (no TR-GMP) Qwen3-4B s${SPARSITY_PCT} kl_budget=${KL_BUDGET} lr=${LR} pgd_interval=${PGD_INTERVAL} lr_scheduler=${LR_SCHEDULER} steps=${STEPS} saliency=${SALIENCY} -- 1xB200 single-GPU, vLLM in-process (OT80/FW20) ==="
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
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_use_fsdp=false \
    --steps=${STEPS} \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=${SALIENCY} \
    --gmp_pruning_scope=${PRUNING_SCOPE} \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=${GRAD_CKPT} \
    --gmp_kl_chunk_size=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=${KD_ONLY} \
    --kd_nsamples=${KD_NSAMPLES} \
    --gmp_ntp_lambda=${NTP_LAMBDA} \
    --gmp_kd_lambda=${KD_LAMBDA} \
    --gmp_onpolicy_kd_lambda=${OPKD_LAMBDA} \
    --gmp_onpolicy_kd_interval=${ROLLOUT_INTERVAL} \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=${VLLM_GPU_MEM} \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=false \
    --gmp_pruning_end_ratio=0.0 \
    --gmp_pgd=true \
    --gmp_pgd_grow_to_target=true \
    --gmp_pgd_kl_budget=${KL_BUDGET} \
    --gmp_pgd_kl_calib_size=${CALIB_SIZE} \
    --gmp_pgd_interval=${PGD_INTERVAL} \
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
    --run_name_suffix="pgd_grow2target_klbudget${KL_BUDGET}_lr${LR}_pgdi${PGD_INTERVAL}_${PRUNING_SCOPE}scope_b200"

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
