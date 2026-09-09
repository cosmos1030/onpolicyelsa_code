#!/bin/bash
# SINGLE-GPU fork of gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu.sh.
#
# WHY IT EXISTS: 8B PGD jobs previously only ran as 2-GPU FSDP, so a GPU whose
# pair-partner was busy could not be used at all -- one sat idle for hours the
# moment a 1-GPU ALPS+SFT baseline and a 2-GPU PGD job ended out of step. This
# gives the scheduler an 8B job shape that fits a single stranded GPU.
#
# EQUIVALENT BY CONSTRUCTION to the 2-GPU recipe, which is exactly the property
# the OPKD sharding fix established (pool scales with world_size, ranks
# partition it):
#     2 GPU: global batch 1 x grad_accum 4 x ws 2 = 8 | pool 32*4*2 = 256
#     1 GPU: global batch 1 x grad_accum 8 x ws 1 = 8 | pool 32*8*1 = 256
# Same global batch, same unique rollouts per refill window, same 8 distinct
# sequences per step. The ALPS+SFT baselines we compare against are themselves
# single-GPU with grad_accum=8, so this is the shape that has always been on the
# other side of the comparison.
#
# Two env differences from the FSDP script, both mandatory on this path:
#   * PYTORCH_CUDA_ALLOC_CONF is NOT set. vLLM's CuMemAllocator (sleep mode) is
#     built in-process here and hard-asserts against expandable_segments; the
#     FSDP script can set it because its vLLM is a separate sidecar process.
#   * gmp_opkd_vllm_gpu_mem defaults to 0.15, the value the single-GPU ALPS+SFT
#     runs use (measured 9.93 GiB KV cache / 4517 blocks at rollout length 512),
#     not the FSDP sidecar's 0.20.
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
# 512, not 256: every 8B PGD run in this project passed 512 explicitly, and the
# 4B launchers already default to 512 -- leaving the default at 256 meant the
# ALPS+SFT baselines silently trained on HALF the on-policy KD tokens per
# rollout that the method they are compared against used (same 256 rollouts per
# refill window either way, but 256 vs 512 tokens each). Do not lower it back
# without re-running both sides.
OPD_GEN_LEN=${4:-512}
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
VLLM_GPU_MEM=${20:-0.15}  # gmp_opkd_vllm_gpu_mem -- see gmp_pgd_klgate_qwen3_8b_fsdp2gpu.sh header for tuning history

NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
DATA_PATH="$DATA_PATH_ARG"
MODEL="Qwen/Qwen3-8B"

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
PYTHON=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

OPD_PROMPT_PATH="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

JOB_TAG="gmp_pgd_grow_8b_1gpu_b200_s${SPARSITY_PCT}_lr${LR}_klb${KL_BUDGET}_pgdi${PGD_INTERVAL}"
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

$PYTHON -u main.py \
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
    --gmp_pgd_jump_to_target=${JUMP:-false} \
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
