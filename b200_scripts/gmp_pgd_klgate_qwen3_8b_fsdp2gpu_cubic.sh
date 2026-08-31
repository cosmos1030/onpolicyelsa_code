#!/bin/bash
# Cubic-growth-schedule ablation fork of gmp_pgd_klgate_qwen3_8b_fsdp2gpu.sh
# (this folder), for the same self-KL-vs-trust-region comparison already run
# on the log_cluster A100 side (elsa/scripts/slurm_gmp_pgd_klgate_qwen3_1.7b_cubic.sh
# / _4b_cubic.sh) but at 8B scale here on B200.
#
# Same recipe as the base FSDP2 script (2xB200, vLLM shares GPU0,
# gmp_opkd_vllm_gpu_mem=0.20, capped PGD every PGD_INTERVAL steps) EXCEPT
# mask growth: instead of trust-region KL-gated growth (--gmp_tr_enabled=true),
# uses a fixed cubic ramp that reaches final_sparsity by step
# STEPS*PRUNING_END_RATIO regardless of self-KL, then freezes the mask for
# the remaining steps (capped PGD keeps running throughout on its own
# PGD_INTERVAL cadence either way). --gmp_cubic_log_kl=true logs
# cubic/kl_before_after at every cubic mask-update boundary via the SAME
# _compute_tr_kl primitive TR-GMP's own gating uses, purely diagnostic (does
# not gate/reject anything) -- lets you compare how far the forced cubic
# schedule's growth steps would have violated the trust-region budget that
# TR-GMP enforces, on the exact wandb key/analysis already used for the 1.7B/
# 4B cubic ablation runs.
#
# **This container has no SLURM** -- run directly with `bash`. Machine-local
# launcher (paths under /NHNHOME/log-postech/doyoonkim/).
#
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash b200_scripts/gmp_pgd_klgate_qwen3_8b_fsdp2gpu_cubic.sh \
#   <SPARSITY> <KL_BUDGET> <KL_THRESHOLD> [MASTER_PORT] [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [POST_TARGET_STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [SPARSITY_TYPE] [L1_LAMBDA] [ROLLOUT_INTERVAL] [KD_NSAMPLES] [CALIB_SIZE] [DEBUG_IMPORTANCE_HIST] [PGD_INTERVAL] [PGD_POST_TARGET_ONLY] [PRUNING_END_RATIO]
# e.g. (S50, matched lr to the existing 8B S50 capped entry yy92hmwi):
#   CUDA_VISIBLE_DEVICES=0,1 bash b200_scripts/gmp_pgd_klgate_qwen3_8b_fsdp2gpu_cubic.sh \
#     0.5 0.02 0.02 29500 512 32 cosine 2048 0 5e-5 "$OT3_DATA" 8192 true reasoning_qwen3_8b_nostrip8192 \
#     fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8 false 0.5
set -e

SPARSITY=${1:?"Usage: <SPARSITY> <KL_BUDGET> <KL_THRESHOLD> [MASTER_PORT] ..."}
KL_BUDGET=${2:?"Usage: <SPARSITY> <KL_BUDGET> <KL_THRESHOLD> [MASTER_PORT] ..."}
KL_THRESHOLD=${3:-0.02}
MASTER_PORT=${4:-29500}
OPD_GEN_LEN=${5:-256}
MASK_INTERVAL=${6:-32}
LR_SCHEDULER=${7:-cosine}
STEPS=${8:-2048}
POST_TARGET_STEPS=${9:-0}
LR=${10:-1e-4}
DATA_PATH_ARG=${11:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${12:-8192}
GRAD_CKPT=${13:-true}
WANDB_PROJECT=${14:-reasoning_qwen3_8b_nostrip8192}
SALIENCY=${15:-fisher}
PRUNING_SCOPE=${16:-global}
LOSS_WEIGHTS=${17:-0.33,0.33,0.33}  # NTP,KD,OPKD
SPARSITY_TYPE=${18:-unstructured}   # unstructured | 2:4 | 4:8
L1_LAMBDA=${19:-0.0}
ROLLOUT_INTERVAL=${20:-${MASK_INTERVAL}}
KD_NSAMPLES=${21:-0}
CALIB_SIZE=${22:-4}
DEBUG_IMPORTANCE_HIST=${23:-false}
PGD_INTERVAL=${24:-1}
PGD_POST_TARGET_ONLY=${25:-false}
PRUNING_END_RATIO=${26:-0.5}  # gmp_pruning_end_ratio -- fixed cubic ramp reaches final_sparsity by step STEPS*this ratio (default 0.5 -> step 1024/2048); mask frozen after that, PGD keeps running capped every PGD_INTERVAL steps regardless.

NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
DATA_PATH="$DATA_PATH_ARG"
MODEL="Qwen/Qwen3-8B"

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
TORCHRUN=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/torchrun

OPD_PROMPT_PATH="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

JOB_TAG="gmp_pgd_klgate_8b_fsdp2_b200_cubic_endratio${PRUNING_END_RATIO}_s${SPARSITY_PCT}_${SPARSITY_TYPE//:/}_lr${LR}_kl${KL_THRESHOLD}_klb${KL_BUDGET}_mi${MASK_INTERVAL}_pgdi${PGD_INTERVAL}${PGD_POST_TARGET_ONLY:+_pto${PGD_POST_TARGET_ONLY}}"
LOCAL_JOB_BASE="/NHNHOME/log-postech/doyoonkim/logs/${JOB_TAG}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)
export WANDB_API_KEY=$(cat /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key)
# FSDP-sidecar vLLM (separate subprocess, not in-process) is fine with
# expandable_segments -- only the single-GPU in-process vLLM path
# (gmp_pgd_klgate_qwen3_4b.sh) needs it left unset. See that script's header.
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

echo "=== TR-GMP NTP+KD+OPKD(${NTP_LAMBDA}/${KD_LAMBDA}/${OPKD_LAMBDA}) Qwen3-8B s${SPARSITY_PCT} ${SPARSITY_TYPE} CUBIC growth (end_ratio=${PRUNING_END_RATIO}) + PGD-KL-budget(self-KL per step<=${KL_BUDGET}, pgd_interval=${PGD_INTERVAL}, post_target_only=${PGD_POST_TARGET_ONLY}) lr=${LR} mask_interval=${MASK_INTERVAL} rollout_interval=${ROLLOUT_INTERVAL} lr_scheduler=${LR_SCHEDULER} steps=${STEPS} saliency=${SALIENCY} -- 2xB200 FSDP, vLLM sharing GPU0 of this pair, master_port=${MASTER_PORT} ==="
echo "NODE=$(hostname)  MODEL=$MODEL  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/elsa

$TORCHRUN --nproc_per_node=2 --master_port=${MASTER_PORT} main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=${SPARSITY_TYPE} \
    --gmp_l1_lambda=${L1_LAMBDA} \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --steps=${STEPS} \
    --gmp_post_target_steps=${POST_TARGET_STEPS} \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
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
    --gmp_opkd_vllm_gpu_mem=0.20 \
    --gmp_opkd_vllm_gpu_index=0 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=false \
    --gmp_growth_schedule=cubic \
    --gmp_pruning_end_ratio=${PRUNING_END_RATIO} \
    --gmp_cubic_log_kl=true \
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
    --run_name_suffix="cubic_endratio${PRUNING_END_RATIO}_pgd_klbudget${KL_BUDGET}_lr${LR}_mi${MASK_INTERVAL}_ro${ROLLOUT_INTERVAL}_${PRUNING_SCOPE}scope_${SPARSITY_TYPE//:/}_b200fsdp2"

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
