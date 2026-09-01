#!/bin/bash
# Local (non-SLURM) single-GPU B200 adaptation of
# elsa/scripts/slurm_gmp_pgd_klgate_qwen3_4b.sh (which mirrors
# elsa/scripts/slurm_gmp_pgd_klgate_qwen3_1.7b.sh scaled to 4B) for this
# docker container: TR-GMP (trust-region gradual mask growth, dense start)
# NTP+KD+OPKD with the per-step PGD reprojection self-KL-gated instead of
# uncapped (--gmp_pgd_kl_budget bisects how many lowest-importance prune
# candidates to accept each step so self-KL(pre||post-prune) stays within
# budget; revive count == accepted prune count, existing invariant).
#
# Positional args now match the log_cluster SLURM 4B/1.7B scripts exactly
# (same order) so the same sbatch-command set can be replayed here with
# `bash` instead of `sbatch` -- includes gmp_pgd_kl_budget's N:M-aware
# reprojection path (sparsity_type=2:4/4:8), gmp_pgd_interval (decouple PGD's
# per-step cost from mask_interval's growth cadence), and
# gmp_pgd_post_target_only (only let PGD reproject once growth has reached
# final_sparsity). NOTE: earlier revisions of this file said
# gmp_pgd_kl_budget only worked for sparsity_type=unstructured -- that was
# true when this file was first written but gmp_trainer.py has since grown a
# dedicated N:M-aware PGD path (_pgd_nm_pre_target/_pgd_nm_post_target), so
# 2:4/4:8 is fine now.
#
# Single B200 (183GB) runs through main.py's plain single-GPU path
# (--gmp_use_fsdp=false, vLLM built in-process) -- same reasoning as the
# other scripts in this folder. See b200_scripts/README.md before changing
# GPU-count-related flags or PYTORCH_CUDA_ALLOC_CONF (vLLM's CuMemAllocator,
# enable_sleep_mode=True, hard-asserts against expandable_segments:True at
# load_model() time -- leave it unset here, matching this folder's other
# OPKD/OPD launchers, not the log_cluster SLURM scripts' max_split_size_mb
# workaround).
# **This container has no SLURM** -- run directly with `bash`. This
# particular box turned out to have 4x B200 (see run_gmp_pgd_klgate_4b_parallel.sh),
# not the single-GPU default this folder otherwise assumes -- check nvidia-smi
# before assuming single-GPU sequential queuing on a new container.
# Machine-local launcher (paths under /NHNHOME/log-postech/doyoonkim/).
#
# Usage: bash b200_scripts/gmp_pgd_klgate_qwen3_4b.sh <SPARSITY> <KL_BUDGET> <KL_THRESHOLD> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [POST_TARGET_STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [SPARSITY_TYPE] [L1_LAMBDA] [ROLLOUT_INTERVAL] [KD_NSAMPLES] [CALIB_SIZE] [DEBUG_IMPORTANCE_HIST] [PGD_INTERVAL] [PGD_POST_TARGET_ONLY] [REVERSE_KL]
# e.g.: bash b200_scripts/gmp_pgd_klgate_qwen3_4b.sh 0.5 999 0.01 512 32 cosine 2048 0 5e-5 \
#         "$OT3_DATA" 8192 true reasoning_qwen3_4b_nostrip8192 fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8
set -e

SPARSITY=${1:?"Usage: <SPARSITY> <KL_BUDGET> <KL_THRESHOLD> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [POST_TARGET_STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [SPARSITY_TYPE] [L1_LAMBDA] [ROLLOUT_INTERVAL] [KD_NSAMPLES] [CALIB_SIZE] [DEBUG_IMPORTANCE_HIST] [PGD_INTERVAL] [PGD_POST_TARGET_ONLY]"}
KL_BUDGET=${2:?"Usage: <SPARSITY> <KL_BUDGET> <KL_THRESHOLD> ..."}
KL_THRESHOLD=${3:-0.02}
OPD_GEN_LEN=${4:-256}
MASK_INTERVAL=${5:-32}
LR_SCHEDULER=${6:-cosine}
STEPS=${7:-2048}
POST_TARGET_STEPS=${8:-0}
LR=${9:-1e-4}
DATA_PATH_ARG=${10:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${11:-8192}
GRAD_CKPT=${12:-true}
WANDB_PROJECT=${13:-reasoning_qwen3_4b_nostrip8192}
SALIENCY=${14:-fisher}
PRUNING_SCOPE=${15:-global}
LOSS_WEIGHTS=${16:-0.33,0.33,0.33}  # NTP,KD,OPKD -- e.g. 0,0.5,0.5 to drop NTP and split KD/OPKD evenly
SPARSITY_TYPE=${17:-unstructured}   # unstructured | 2:4 | 4:8
L1_LAMBDA=${18:-0.0}                # gmp_l1_lambda -- structured-L1 pre-shrink for N:M endgame (0=off)
ROLLOUT_INTERVAL=${19:-${MASK_INTERVAL}}  # gmp_onpolicy_kd_interval -- defaults to mask_interval
KD_NSAMPLES=${20:-0}  # gmp_kd_nsamples -- 0 = full dataset
CALIB_SIZE=${21:-4}   # gmp_pgd_kl_calib_size
DEBUG_IMPORTANCE_HIST=${22:-false}  # gmp_pgd_debug_importance_hist -- diagnostic only, ~0.6s/step amortized
PGD_INTERVAL=${23:-1}  # gmp_pgd_interval -- run PGD's reprojection only every Nth step
PGD_POST_TARGET_ONLY=${24:-false}  # gmp_pgd_post_target_only -- PGD only fires once growth reaches final_sparsity
REVERSE_KL=${25:-false}  # gmp_onpolicy_reverse_kl -- reverse KL D(S||T) for on-policy KD instead of forward KL D(T||S) (default)

NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
DATA_PATH="$DATA_PATH_ARG"
MODEL="Qwen/Qwen3-4B"

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
PYTHON=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

OPD_PROMPT_PATH="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

JOB_TAG="gmp_pgd_klgate_4b_b200_s${SPARSITY_PCT}_${SPARSITY_TYPE//:/}_lr${LR}_kl${KL_THRESHOLD}_klb${KL_BUDGET}_mi${MASK_INTERVAL}_pgdi${PGD_INTERVAL}${PGD_POST_TARGET_ONLY:+_pto${PGD_POST_TARGET_ONLY}}${REVERSE_KL:+_rkl${REVERSE_KL}}"
LOCAL_JOB_BASE="/NHNHOME/log-postech/doyoonkim/logs/${JOB_TAG}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)
export WANDB_API_KEY=$(cat /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key)
# expandable_segments left UNSET on purpose -- see header note / README.
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/torchinductor
export VLLM_CACHE_ROOT=/NHNHOME/log-postech/doyoonkim/.cache/vllm
export HF_HOME=/NHNHOME/log-postech/doyoonkim/.cache/huggingface
# Deliberately NOT setting HF_DATASETS_OFFLINE/TRANSFORMERS_OFFLINE -- this
# container has internet and needs it for eval datasets not already cached.
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
# Disables vLLM's background usage-reporting thread -- observed on the
# log_cluster A100 runs to cause a rare interpreter-level crash deep into
# training, always right at a vLLM wake_up() call. Not needed here either.
export VLLM_NO_USAGE_STATS=1

echo "=== TR-GMP NTP+KD+OPKD(${NTP_LAMBDA}/${KD_LAMBDA}/${OPKD_LAMBDA}) Qwen3-4B s${SPARSITY_PCT} ${SPARSITY_TYPE} PGD-KL-budget(self-KL per step<=${KL_BUDGET}, tr_kl=${KL_THRESHOLD}, pgd_interval=${PGD_INTERVAL}, post_target_only=${PGD_POST_TARGET_ONLY}) lr=${LR} mask_interval=${MASK_INTERVAL} rollout_interval=${ROLLOUT_INTERVAL} lr_scheduler=${LR_SCHEDULER} steps=${STEPS} post_target_steps=${POST_TARGET_STEPS} saliency=${SALIENCY} -- 1xB200 single-GPU, vLLM in-process ==="
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
    --gmp_l1_lambda=${L1_LAMBDA} \
    --do_gmp=true \
    --gmp_use_fsdp=false \
    --steps=${STEPS} \
    --gmp_post_target_steps=${POST_TARGET_STEPS} \
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
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=${KD_ONLY} \
    --kd_nsamples=${KD_NSAMPLES} \
    --gmp_ntp_lambda=${NTP_LAMBDA} \
    --gmp_kd_lambda=${KD_LAMBDA} \
    --gmp_onpolicy_kd_lambda=${OPKD_LAMBDA} \
    --gmp_onpolicy_kd_interval=${ROLLOUT_INTERVAL} \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_pgd=true \
    --gmp_pgd_kl_budget=${KL_BUDGET} \
    --gmp_pgd_kl_calib_size=${CALIB_SIZE} \
    --gmp_pgd_debug_importance_hist=${DEBUG_IMPORTANCE_HIST} \
    --gmp_pgd_interval=${PGD_INTERVAL} \
    --gmp_pgd_post_target_only=${PGD_POST_TARGET_ONLY} \
    --gmp_onpolicy_reverse_kl=${REVERSE_KL} \
    --gmp_pgd_skip_growth_step=true \
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
    --run_name_suffix="pgd_klbudget${KL_BUDGET}_skipgrowth_lr${LR}_mi${MASK_INTERVAL}_ro${ROLLOUT_INTERVAL}_kl${KL_THRESHOLD}_${PRUNING_SCOPE}scope_${SPARSITY_TYPE//:/}${REVERSE_KL:+_rkl${REVERSE_KL}}_b200"

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
