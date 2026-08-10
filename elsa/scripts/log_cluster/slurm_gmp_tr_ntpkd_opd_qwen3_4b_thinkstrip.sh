#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opd_4b_ts
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/tr_ntpkd_opd_4b_ts_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPD (0.33/0.33/0.33), Qwen3-4B, dense -> gradual Fisher-saliency
# pruning to target sparsity via trust-region mask growth, on the new
# THINKSTRIP-200K dataset (cosmos1030/ot3-fineweb-200k-qwen3-thinkstrip) instead
# of PLAIN-200K -- see elsa/data/DATASETS.md. THINKSTRIP strips <think>...</think>
# down to the post-think write-up whenever a conversation would otherwise exceed
# seqlen=2048, reducing truncation from 80.3% (PLAIN-200K) to 48.5% and avoiding
# PLAIN-200K's failure mode of losing the final \boxed{} answer to truncation.
#
# Same recipe as slurm_gmp_tr_ntpkd_opd_qwen3_4b.sh (mask_interval=32 fixed,
# confirmed on 2026-08-08 to beat mask_interval=8 at every sparsity tested --
# see SESSION_HANDOFF.md), plus --gmp_post_target_steps=0 (stop as soon as the
# TR trust-region growth reaches final_sparsity instead of continuing with a
# frozen mask for the full 2048-step budget). LR and KL_THRESHOLD are the
# sweep axes here (mask_interval intentionally left fixed, not swept).
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b_thinkstrip.sh <SPARSITY> <LR> <KL_THRESHOLD> [LR_SCHEDULER] [MASK_INTERVAL]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b_thinkstrip.sh 0.5 1e-4 0.01
#       sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b_thinkstrip.sh 0.5 5e-5 0.02

SPARSITY=${1:?"Usage: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b_thinkstrip.sh <SPARSITY> <LR> <KL_THRESHOLD> [LR_SCHEDULER] [MASK_INTERVAL]"}
LR=${2:?"Usage: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b_thinkstrip.sh <SPARSITY> <LR> <KL_THRESHOLD> [LR_SCHEDULER] [MASK_INTERVAL]"}
KL_THRESHOLD=${3:?"Usage: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b_thinkstrip.sh <SPARSITY> <LR> <KL_THRESHOLD> [LR_SCHEDULER] [MASK_INTERVAL]"}
LR_SCHEDULER=${4:-cosine}
MASK_INTERVAL=${5:-32}

MODEL="Qwen/Qwen3-4B"
REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
DATA_PATH="${REPO_ROOT}/data/ot3_fineweb_200k_qwen3_thinkstrip.jsonl"
OPD_PROMPT_PATH="${REPO_ROOT}/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

LOCAL_JOB_BASE="/tmp/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/models"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_HOME=/home/shared/huggingface
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "=== TR-GMP NTP+KD+OPD (0.33/0.33/0.33) Qwen3-4B s${SPARSITY} lr=${LR} kl=${KL_THRESHOLD} mask_interval=${MASK_INTERVAL} (THINKSTRIP-200K) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$MODEL  DATA=$DATA_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd "$REPO_ROOT"

python main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --steps=2048 \
    --gmp_post_target_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --seqlen=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_kd_only=false \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_saliency=fisher \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_use_fsdp=false \
    --gmp_save_path="${REPO_ROOT}/models" \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b \
    --seed=42

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
