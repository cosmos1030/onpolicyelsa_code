#!/bin/bash
# PGD-driven growth (no separate TR-GMP/cubic growth mechanism) for this B200
# container's 2xGPU FSDP path -- fork of gmp_pgd_klgate_qwen3_8b_fsdp2gpu.sh
# (this folder) with growth itself replaced by --gmp_pgd_grow_to_target=true,
# mirroring gmp_pgd_grow_to_target_qwen3_4b.sh's single-GPU recipe: _pgd_desired
# targets final_sparsity directly (instead of matching current keep-count),
# and revive is no longer forced equal to prune (revive saturates at
# min(k, revive_cand) while prune keeps going up to k), so the self-KL-gated
# bisection alone drives net sparsity growth toward target, at whatever pace
# --gmp_pgd_kl_budget allows. Once current sparsity reaches target,
# prune_cand/revive_cand naturally converge and it degrades to pure
# polish/maintenance -- no separate post-target branch needed (unlike the
# N:M pre/post-target split gmp_pgd_klgate_qwen3_8b_fsdp2gpu_cubic.sh's N:M
# path uses).
#
# --gmp_pruning_end_ratio=0.0 (with --gmp_tr_enabled=false) disables the
# schedule-driven growth path entirely -- PGD, gated separately by its own
# step%pgd_interval==0 check, is left as the ONLY thing that can move the
# mask. unstructured only (not yet supported for sparsity_type=2:4/N:M --
# would additionally need the desired-mask target to respect per-group
# structural caps; see gmp_pgd_grow_to_target_qwen3_4b.sh's header and
# elsa/main.py's flag docstring).
#
# This is the SAME unstructured gmp_pgd_kl_budget branch our 8B FSDP2
# unstructured cubic-ablation runs (S50/S60/S70) already exercised
# successfully under FSDP -- grow_to_target only changes how _pgd_desired is
# computed within that branch, not the FSDP collective/broadcast machinery
# around it. The N:M OOM hit during the 2:4 cubic wave2 run (92eeb8a's
# _pgd_kl_at_nm_post) is a DIFFERENT code path (N:M pre/post-target) that
# this script never touches, since sparsity_type is hardcoded to
# unstructured below.
#
# 2xB200 FSDP, vLLM sidecar sharing GPU0 of this job's pair
# (--gmp_opkd_vllm_gpu_index=0), same skeleton as
# gmp_pgd_klgate_qwen3_8b_fsdp2gpu.sh -- see that script's header for the
# vLLM gpu_mem tuning history (0.20 clears vLLM's own ~32GiB weights+overhead
# with thin-but-workable KV-cache room for unstructured jobs).
#
# **This container has no SLURM** -- run directly with `bash`. Machine-local
# launcher (paths under /NHNHOME/log-postech/doyoonkim/).
#
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash b200_scripts/gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu.sh \
#   <SPARSITY> <KL_BUDGET> [MASTER_PORT] [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [ROLLOUT_INTERVAL] [KD_NSAMPLES] [CALIB_SIZE] [PGD_INTERVAL] [VLLM_GPU_MEM]
# e.g. (S50, matched lr/kl to the existing 4B grow_to_target S50 launch):
#   CUDA_VISIBLE_DEVICES=0,1 bash b200_scripts/gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu.sh \
#     0.5 0.02 29500 512 32 cosine 2048 5e-5 "$OT3_DATA" 8192 true reasoning_qwen3_8b_nostrip8192 \
#     fisher global 0.33,0.33,0.33 32 0 4 8 0.20
set -e

SPARSITY=${1:?"Usage: <SPARSITY> <KL_BUDGET> [MASTER_PORT] ..."}
KL_BUDGET=${2:?"Usage: <SPARSITY> <KL_BUDGET> [MASTER_PORT] ..."}
MASTER_PORT=${3:-29500}
OPD_GEN_LEN=${4:-256}
MASK_INTERVAL=${5:-32}
LR_SCHEDULER=${6:-cosine}
STEPS=${7:-2048}
LR=${8:-1e-4}
DATA_PATH_ARG=${9:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${10:-8192}
GRAD_CKPT=${11:-true}
WANDB_PROJECT=${12:-reasoning_qwen3_8b_nostrip8192}
SALIENCY=${13:-fisher}
PRUNING_SCOPE=${14:-global}
LOSS_WEIGHTS=${15:-0.33,0.33,0.33}  # NTP,KD,OPKD
ROLLOUT_INTERVAL=${16:-${MASK_INTERVAL}}
KD_NSAMPLES=${17:-0}
CALIB_SIZE=${18:-4}
PGD_INTERVAL=${19:-8}  # gmp_pgd_interval -- also the effective growth cadence in this mode
VLLM_GPU_MEM=${20:-0.20}  # gmp_opkd_vllm_gpu_mem -- see gmp_pgd_klgate_qwen3_8b_fsdp2gpu.sh header for tuning history

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

JOB_TAG="gmp_pgd_grow_8b_fsdp2_b200_s${SPARSITY_PCT}_lr${LR}_klb${KL_BUDGET}_pgdi${PGD_INTERVAL}"
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
# (gmp_pgd_grow_to_target_qwen3_4b.sh) needs it left unset.
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

echo "=== PGD-driven growth (no TR-GMP) Qwen3-8B s${SPARSITY_PCT} kl_budget=${KL_BUDGET} lr=${LR} pgd_interval=${PGD_INTERVAL} lr_scheduler=${LR_SCHEDULER} steps=${STEPS} saliency=${SALIENCY} -- 2xB200 FSDP, vLLM sharing GPU0 of this pair, master_port=${MASTER_PORT} ==="
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
    --sparsity_type=unstructured \
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
    --gmp_ckpt_every_steps=${CKPT_EVERY:-0} --gmp_ckpt_dir="${CKPT_DIR:-}" --gmp_resume_from="${RESUME_FROM:-}" \
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
    --run_name_suffix="pgd_grow2target_klbudget${KL_BUDGET}_lr${LR}_pgdi${PGD_INTERVAL}_${PRUNING_SCOPE}scope_b200fsdp2"

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
