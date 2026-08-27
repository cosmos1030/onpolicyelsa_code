#!/bin/bash
# Qwen3-8B FSDP 2-GPU + PGD-KL-gate launcher for this B200 container, filling
# a gap: the runs behind the dashboard's "yy92hmwi/bmzorogu/vez1mi9f/t5x5gwpr"
# 8B FSDP-2GPU PGD kl_budget results (2026-08-25) were launched ad-hoc and
# never got a script committed to git -- this reconstructs that recipe as a
# proper parameterized launcher, grafting elsa/scripts/log_cluster/
# slurm_gmp_tr_ntpkd_opd_qwen3_8b_fsdp2gpu.sh's FSDP/vLLM-sidecar skeleton
# (single-H200 8B OOM'd at step 1 inside the full-vocab KD KL loss, ~136GB
# peak -- FSDP shards student weights/grad/optimizer across 2 GPUs instead)
# onto gmp_pgd_klgate_qwen3_4b.sh's PGD flag set (kl_budget, pgd_interval,
# post_target_only, N:M support via SPARSITY_TYPE).
#
# vLLM (OPD rollouts) shares GPU0 of this job's 2-GPU pair
# (--gmp_opkd_vllm_gpu_index=0, tp_size=1) instead of a dedicated 3rd GPU --
# same reasoning as the log_cluster script: TP=2 deadlocks the moment another
# live torch.distributed pg (this job's FSDP group) exists on the same
# physical GPUs, and this container has no spare 3rd GPU per job anyway once
# two 8B jobs run concurrently (see run_gmp_pgd_klgate_8b_fsdp2gpu_parallel.sh
# -- 4 GPUs total / 2 per job = 2 concurrent jobs, CUDA_VISIBLE_DEVICES-pinned
# to disjoint pairs, each job passing its OWN --master_port).
#
# PGD's self-KL bisection forward pass is an FSDP collective (per-layer
# all_gather); a real fix (2026-08-25, gmp_trainer.py) broadcasts the
# resulting KL scalar from rank0 so bisection takes the same branch on every
# rank -- without it, rank-local floating-point drift near the budget
# boundary could make ranks call a different number of forwards and the FSDP
# collective count desyncs into a deadlock (reproduced live: two independent
# 8B FSDP jobs both hung at the same step). Already applied in this repo's
# gmp_trainer.py -- nothing to do here, just don't strip --gmp_pgd_kl_budget
# assuming it's a single-GPU-only feature.
#
# Global batch = nproc_per_node(2) x gmp_batch_size(1) x gmp_grad_accum(4) = 8,
# matching the reference runs ("2×1×4").
#
# gmp_opkd_vllm_gpu_mem=0.25 (not the log_cluster script's 0.15): first live
# attempt at 0.15 (2026-08-27, s50/s60 concurrent) crashed both jobs inside
# vLLM's KV-cache init -- "total_gpu_memory(178.35GiB) x 0.15 = 26.75GiB"
# budget was already short of vLLM's own weights+overhead (~32GB) by
# ~5.5GiB, before any KV cache. Oddly, the next batch (s70/s50_24, same
# flags, same host) launched cleanly -- looked timing-sensitive (vLLM's
# memory profiling racing the co-resident FSDP rank0's initial unsharded
# weight load on GPU0), not a hard wall, but bumped to 0.25 for headroom
# rather than relying on that race resolving favorably every time.


#
# **This container has no SLURM** -- run directly with `bash`. Machine-local
# launcher (paths under /NHNHOME/log-postech/doyoonkim/).
#
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash b200_scripts/gmp_pgd_klgate_qwen3_8b_fsdp2gpu.sh \
#   <SPARSITY> <KL_BUDGET> <KL_THRESHOLD> [MASTER_PORT] [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [POST_TARGET_STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [SPARSITY_TYPE] [L1_LAMBDA] [ROLLOUT_INTERVAL] [KD_NSAMPLES] [CALIB_SIZE] [DEBUG_IMPORTANCE_HIST] [PGD_INTERVAL] [PGD_POST_TARGET_ONLY]
# e.g.: CUDA_VISIBLE_DEVICES=0,1 bash b200_scripts/gmp_pgd_klgate_qwen3_8b_fsdp2gpu.sh \
#         0.5 999 0.01 29500 512 32 cosine 2048 0 5e-5 "$OT3_DATA" 8192 true reasoning_qwen3_8b_nostrip8192 \
#         fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8
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

JOB_TAG="gmp_pgd_klgate_8b_fsdp2_b200_s${SPARSITY_PCT}_${SPARSITY_TYPE//:/}_lr${LR}_kl${KL_THRESHOLD}_klb${KL_BUDGET}_mi${MASK_INTERVAL}_pgdi${PGD_INTERVAL}${PGD_POST_TARGET_ONLY:+_pto${PGD_POST_TARGET_ONLY}}"
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
# (gmp_pgd_klgate_qwen3_4b.sh) needs it left unset. See that script's header
# and elsa/scripts/log_cluster/slurm_gmp_tr_ntpkd_opd_qwen3_8b_fsdp2gpu.sh.
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

echo "=== TR-GMP NTP+KD+OPKD(${NTP_LAMBDA}/${KD_LAMBDA}/${OPKD_LAMBDA}) Qwen3-8B s${SPARSITY_PCT} ${SPARSITY_TYPE} PGD-KL-budget(self-KL per step<=${KL_BUDGET}, tr_kl=${KL_THRESHOLD}, pgd_interval=${PGD_INTERVAL}, post_target_only=${PGD_POST_TARGET_ONLY}) lr=${LR} mask_interval=${MASK_INTERVAL} rollout_interval=${ROLLOUT_INTERVAL} lr_scheduler=${LR_SCHEDULER} steps=${STEPS} saliency=${SALIENCY} -- 2xB200 FSDP, vLLM sharing GPU0 of this pair, master_port=${MASTER_PORT} ==="
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
    --gmp_opkd_vllm_gpu_mem=0.25 \
    --gmp_opkd_vllm_gpu_index=0 \
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
    --run_name_suffix="pgd_klbudget${KL_BUDGET}_skipgrowth_lr${LR}_mi${MASK_INTERVAL}_ro${ROLLOUT_INTERVAL}_kl${KL_THRESHOLD}_${PRUNING_SCOPE}scope_${SPARSITY_TYPE//:/}_b200fsdp2"

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
