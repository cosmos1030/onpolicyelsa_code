#!/bin/bash
# N:M (2:4) twin of gmp_pgd_grow_to_target_qwen3_4b.sh (this folder) -- same
# B200 single-GPU/in-process-vLLM conventions, --sparsity_type=2:4 instead
# of unstructured.
#
# N:M support in --gmp_pgd_grow_to_target: the 2:4 pattern is treated as a
# constraint on the FINAL target mask only, not on intermediate masks (a
# group is free to sit at any dead-count mid-training) -- three self-KL-
# gated phases per PGD step (overshoot-prune-only / undershoot-revive-only /
# finished-group atomic-paired-swap), combined via ONE joint bisection over
# a shared fraction alpha so the whole step's self-KL is checked against the
# TRUE pre-step state, not three independently-budgeted sub-checks. Two
# runtime invariants are checked EVERY step and raise RuntimeError on
# violation: (1) no group's |alive-prune_n| ever increases, (2) the
# whole-step D_KL(before||after), re-measured on the exact applied
# candidate (not a re-drawn one, since _pgd_topk_mask_from_vals/
# _pgd_topk_groups_from_scores use torch.rand_like tie-breaking and a
# re-drawn candidate at the same alpha is NOT the same mask), must be
# <= budget+1e-6.
#
# Validated on the log_cluster A100 side (1.7B, 100-step smoke test, job
# 828652) both in-loop (0 invariant violations across the whole run) and by
# an INDEPENDENT post-hoc check directly on the saved safetensors (outside
# the training code, no shared functions): 196/197 prunable tensors at
# exactly 0% violation (every 4-group exactly 2-alive); the one non-zero
# tensor was embed_tokens.weight, outside maskmgr's pruning scope entirely
# (never touched), not a violation. Then launched as 3 production 1.7B 2:4
# runs (lr=1e-4, kl_budget in {0.005, 0.01, 0.02}, ro=32) on log_cluster --
# this is the 4B counterpart for the B200 container.
#
# **This container has no SLURM** -- run directly with `bash`. Machine-local
# launcher (paths under /NHNHOME/log-postech/doyoonkim/).
#
# Usage: bash b200_scripts/gmp_pgd_grow_to_target_qwen3_4b_24.sh <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [ROLLOUT_INTERVAL] [KD_NSAMPLES] [CALIB_SIZE] [PGD_INTERVAL] [VLLM_GPU_MEM]
# e.g. (matching the 1.7B 2:4 production sweep: lr=1e-4, ro=32):
#   bash b200_scripts/gmp_pgd_grow_to_target_qwen3_4b_24.sh 0.02 512 32 cosine 2048 1e-4 "$OT3_DATA" 8192 true reasoning_qwen3_4b_nostrip8192 fisher global 0.33,0.33,0.33 32 0 4 8 0.15
set -e

KL_BUDGET=${1:?"Usage: <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] ..."}
OPD_GEN_LEN=${2:-512}
MASK_INTERVAL=${3:-32}
LR_SCHEDULER=${4:-cosine}
STEPS=${5:-2048}
LR=${6:-1e-4}
DATA_PATH_ARG=${7:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${8:-8192}
GRAD_CKPT=${9:-true}
WANDB_PROJECT=${10:-reasoning_qwen3_4b_nostrip8192}
SALIENCY=${11:-fisher}
PRUNING_SCOPE=${12:-global}
LOSS_WEIGHTS=${13:-0.33,0.33,0.33}  # NTP,KD,OPKD
ROLLOUT_INTERVAL=${14:-${MASK_INTERVAL}}  # gmp_onpolicy_kd_interval -- defaults to mask_interval
KD_NSAMPLES=${15:-0}  # 0 = full dataset (production)
CALIB_SIZE=${16:-4}   # gmp_pgd_kl_calib_size
PGD_INTERVAL=${17:-8}  # gmp_pgd_interval
VLLM_GPU_MEM=${18:-0.15}  # gmp_opkd_vllm_gpu_mem -- 0.15 sized for OPD_GEN_LEN=256, raise for 512+

NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
DATA_PATH="$DATA_PATH_ARG"

source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac
PYTHON=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

MODEL="Qwen/Qwen3-4B"
OPD_PROMPT_PATH="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

JOB_TAG="gmp_pgd_grow_4b_24_b200_klb${KL_BUDGET}_lr${LR}_pgdi${PGD_INTERVAL}"
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

echo "=== PGD-driven growth (no TR-GMP) Qwen3-4B 2:4 kl_budget=${KL_BUDGET} lr=${LR} pgd_interval=${PGD_INTERVAL} mask_interval=${MASK_INTERVAL} rollout_interval=${ROLLOUT_INTERVAL} lr_scheduler=${LR_SCHEDULER} steps=${STEPS} saliency=${SALIENCY} -- 1xB200 single-GPU, vLLM in-process (OT80/FW20) ==="
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
    --sparsity_ratio=0.5 \
    --sparsity_type=2:4 \
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
    --run_name_suffix="pgd_grow2target_24_klbudget${KL_BUDGET}_lr${LR}_pgdi${PGD_INTERVAL}_${PRUNING_SCOPE}scope_b200"

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
