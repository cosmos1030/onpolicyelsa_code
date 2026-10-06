#!/bin/bash
#SBATCH --job-name=pgd_ablation_pilot
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --exclude=n3,n42,n51,n54,n60,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/pgd_ablation_pilot_%j.out
exec 2>&1

# Short pilot: isolate PGD (with vs without a self-KL trust region, at
# several budget values) from TR-GMP growth entirely, starting from a
# FROZEN ALPS one-shot s70 mask (gmp_fixed_mask=true, gmp_tr_enabled=false)
# instead of growing from dense. Answers "does the self-KL trust region on
# PGD's reprojection matter" without the growth-schedule confound.
#
# Usage: sbatch slurm_pgd_ablation_alps_s70_pilot.sh <ARM> [STEPS] [LR]
#   ARM: plain (no KL gate, fully uncapped desired-mask threshold swap)
#        kl02 / kl002 / kl0002  (gmp_pgd_kl_budget = 0.02 / 0.002 / 0.0002)

ARM=${1:?"Usage: <ARM: plain|kl02|kl002|kl0002> [STEPS] [LR] [PRUNE_OPD] [OPD_GEN_LEN]"}
STEPS=${2:-128}
LR=${3:-5e-5}
PRUNE_OPD=${4:-false}  # gmp_opkd_prune_opd -- Prune-OPD-style token-reliability decay weighting on the on-policy KD loss
OPD_GEN_LEN=${5:-512}  # gmp_onpolicy_max_new_tokens -- longer rollouts give the monotone-decay weight more room to actually diverge from 1.0 before the sequence ends; the 512-token default barely gives prune_opd a chance to bite

PGD_ENABLED=true
case "$ARM" in
  none)    PGD_ENABLED=false; PGD_KL_BUDGET=0 ;;
  plain)   PGD_KL_BUDGET=0 ;;
  kl02)    PGD_KL_BUDGET=0.02 ;;
  kl002)   PGD_KL_BUDGET=0.002 ;;
  kl0005)  PGD_KL_BUDGET=0.0005 ;;
  kl0002)  PGD_KL_BUDGET=0.0002 ;;
  *) echo "Unknown ARM: $ARM (expected none|plain|kl02|kl002|kl0005|kl0002)"; exit 1 ;;
esac

ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_1.7b_alps_s70pct"
DENSE_MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== PGD ablation pilot: ARM=${ARM} (pgd_kl_budget=${PGD_KL_BUDGET}) ALPS-s70 frozen mask, no TR growth, steps=${STEPS} lr=${LR} ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$ALPS_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$ALPS_MODEL" \
    --gmp_teacher_model="$DENSE_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=0.7 \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --gmp_tr_enabled=false \
    --steps=${STEPS} \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --seqlen=8192 \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_pgd=${PGD_ENABLED} \
    --gmp_pgd_interval=8 \
    --gmp_pgd_skip_growth_step=true \
    $([ "$PGD_ENABLED" = "true" ] && echo "--gmp_pgd_kl_budget=${PGD_KL_BUDGET} --gmp_pgd_kl_calib_size=4") \
    --gmp_opkd_prune_opd=${PRUNE_OPD} \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=true \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b_nostrip8192 \
    --run_name_suffix="pgd_ablation_pilot_${ARM}_s70_lr${LR}_steps${STEPS}_genlen${OPD_GEN_LEN}_$([ "${PRUNE_OPD}" = "true" ] && echo pruneopd_)" \
    --seed=42

echo "##### END #####"
