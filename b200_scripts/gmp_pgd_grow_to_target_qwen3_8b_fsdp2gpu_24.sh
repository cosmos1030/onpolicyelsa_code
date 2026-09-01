#!/bin/bash
# N:M (2:4) twin of gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu.sh (this
# folder) -- same 2xB200 FSDP skeleton, --sparsity_type=2:4 instead of
# unstructured.
#
# N:M support in --gmp_pgd_grow_to_target: the 2:4 pattern is treated as a
# constraint on the FINAL target mask only, not on intermediate masks (a
# group is free to sit at any dead-count mid-training) -- three self-KL-
# gated phases per PGD step (overshoot-prune-only / undershoot-revive-only /
# finished-group atomic-paired-swap), combined via ONE joint bisection over
# a shared fraction alpha so the whole step's self-KL is checked against the
# TRUE pre-step state once, not three independently-budgeted sub-checks.
# Two runtime invariants are checked EVERY step and raise RuntimeError on
# violation: (1) no group's |alive-prune_n| ever increases, (2) the
# whole-step D_KL(before||after), re-measured on the exact applied
# candidate (not a re-drawn one -- the topk selectors use torch.rand_like
# tie-breaking, so re-deriving at the same alpha silently produces a
# different mask), must stay within budget.
#
# FSDP-specific correctness (this is the part that ONLY matters once you're
# running multi-rank, i.e. exactly this script): the finished-group swap
# pool (_pgd_nm_finished_swap_build) gathers to a FULL, rank-IDENTICAL
# tensor -- unlike the overshoot/undershoot pools, which stay genuinely
# per-rank-local (scattered back after gathering). Two bugs specific to
# that rank-identical case were caught by design review BEFORE ever running
# this under real FSDP (neither would have crashed -- both would have
# silently produced a WRONG result): (1) the finished-candidate count must
# NOT be all_reduced a second time (it's already global after the gather;
# an extra all_reduce(SUM) multiplies it by world_size), and (2) the
# tie-breaking random draw (torch.rand) must be seeded identically across
# ranks (seed=step) -- otherwise every rank independently draws different
# random numbers on IDENTICAL input data and can select DIFFERENT groups,
# desyncing which coordinates each FSDP rank thinks are pruned/revived.
# Validated on a 2xRTX3090 FSDP smoke test (job 829910, Qwen3-0.6B, 40
# steps): both ranks logged byte-identical n_finished_cand/k_finished/
# kl_final at every step, 0 invariant violations, clean exit.
#
# 2xB200 FSDP, vLLM sidecar sharing GPU0 of this job's pair
# (--gmp_opkd_vllm_gpu_index=0), same skeleton as
# gmp_pgd_klgate_qwen3_8b_fsdp2gpu_cubic.sh's N:M path -- see that script's
# header for the vLLM gpu_mem tuning history / the 2:4 OOM headroom shave.
#
# **This container has no SLURM** -- run directly with `bash`. Machine-local
# launcher (paths under /NHNHOME/log-postech/doyoonkim/).
#
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash b200_scripts/gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu_24.sh \
#   <KL_BUDGET> [MASTER_PORT] [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [ROLLOUT_INTERVAL] [KD_NSAMPLES] [CALIB_SIZE] [PGD_INTERVAL] [VLLM_GPU_MEM]
# e.g. (matching the 1.7B 2:4 production sweep: lr=1e-4, ro=32):
#   CUDA_VISIBLE_DEVICES=0,1 bash b200_scripts/gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu_24.sh \
#     0.02 29500 512 32 cosine 2048 1e-4 "$OT3_DATA" 8192 true reasoning_qwen3_8b_nostrip8192 \
#     fisher global 0.33,0.33,0.33 32 0 4 8 0.20
set -e

KL_BUDGET=${1:?"Usage: <KL_BUDGET> [MASTER_PORT] ..."}
MASTER_PORT=${2:-29500}
OPD_GEN_LEN=${3:-256}
MASK_INTERVAL=${4:-32}
LR_SCHEDULER=${5:-cosine}
STEPS=${6:-2048}
LR=${7:-1e-4}
DATA_PATH_ARG=${8:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${9:-8192}
GRAD_CKPT=${10:-true}
WANDB_PROJECT=${11:-reasoning_qwen3_8b_nostrip8192}
SALIENCY=${12:-fisher}
PRUNING_SCOPE=${13:-global}
LOSS_WEIGHTS=${14:-0.33,0.33,0.33}  # NTP,KD,OPKD
ROLLOUT_INTERVAL=${15:-${MASK_INTERVAL}}
KD_NSAMPLES=${16:-0}
CALIB_SIZE=${17:-4}
PGD_INTERVAL=${18:-8}
VLLM_GPU_MEM=${19:-0.20}  # gmp_opkd_vllm_gpu_mem -- see gmp_pgd_klgate_qwen3_8b_fsdp2gpu_cubic.sh header for the 2:4-specific OOM headroom shave; raise the unstructured-tuned 0.20 default down if this OOMs at post-target maintenance.

NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
DATA_PATH="$DATA_PATH_ARG"
MODEL="Qwen/Qwen3-8B"

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
TORCHRUN=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/torchrun

OPD_PROMPT_PATH="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

JOB_TAG="gmp_pgd_grow_8b_24_fsdp2_b200_klb${KL_BUDGET}_lr${LR}_pgdi${PGD_INTERVAL}"
LOCAL_JOB_BASE="/NHNHOME/log-postech/doyoonkim/logs/${JOB_TAG}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat /NHNHOME/log-postech/doyoonkim/secrets/hf_token)
export WANDB_API_KEY=$(cat /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key)
# FSDP-sidecar vLLM (separate subprocess, not in-process) is fine with
# expandable_segments -- only the single-GPU in-process vLLM path needs it
# left unset.
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

echo "=== PGD-driven growth (no TR-GMP) Qwen3-8B 2:4 kl_budget=${KL_BUDGET} lr=${LR} pgd_interval=${PGD_INTERVAL} lr_scheduler=${LR_SCHEDULER} steps=${STEPS} saliency=${SALIENCY} -- 2xB200 FSDP, vLLM sharing GPU0 of this pair, master_port=${MASTER_PORT} ==="
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
    --sparsity_ratio=0.5 \
    --sparsity_type=2:4 \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --steps=${STEPS} \
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
    --gmp_opkd_vllm_gpu_mem=${VLLM_GPU_MEM} \
    --gmp_opkd_vllm_gpu_index=0 \
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
    --run_name_suffix="pgd_grow2target_24_klbudget${KL_BUDGET}_lr${LR}_pgdi${PGD_INTERVAL}_${PRUNING_SCOPE}scope_b200fsdp2"

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
