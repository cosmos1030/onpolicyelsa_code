#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opd_4b_gen
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/tr_ntpkd_opd_4b_gen_%j.out
exec 2>&1

# General-purpose TR-GMP NTP+KD+OPD (0.33/0.33/0.33) launcher for Qwen3-4B on
# this cluster (log_cluster) -- DATA_PATH is a required argument instead of
# hardcoded, so the same script covers PLAIN/THINKSTRIP/RQAT/100pct-OT3 etc.
# Mirrors the mask_interval=32-fixed, gmp_post_target_steps=0 recipe used for
# the THINKSTRIP sweep (2026-08-09, see SESSION_HANDOFF.md).
#
# Usage: sbatch [--partition=A100 --job-name=... --output=...] slurm_gmp_tr_ntpkd_opd_qwen3_4b_general.sh \
#          <SPARSITY> <LR> <KL_THRESHOLD> <DATA_PATH> [OPD_PROMPT_PATH] [MASK_INTERVAL] [LR_SCHEDULER] [SALIENCY] [MODEL]
# e.g. (4B/H200, default): sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b_general.sh 0.5 5e-5 0.01 \
#         /home/doyoonkim/projects/onpolicyelsa_code/elsa/data/ot3_100pct_qwen3_100k.jsonl "" 32 cosine spa
# e.g. (1.7B/A100, override partition via sbatch flag): sbatch --partition=A100 --job-name=tr_ntpkd_opd_1.7b \
#         slurm_gmp_tr_ntpkd_opd_qwen3_4b_general.sh 0.5 5e-5 0.01 \
#         /home/doyoonkim/projects/onpolicyelsa_code/elsa/data/ot3_100pct_qwen3_100k.jsonl \
#         "" 32 cosine spa Qwen/Qwen3-1.7B

SPARSITY=${1:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <DATA_PATH> [OPD_PROMPT_PATH] [MASK_INTERVAL] [LR_SCHEDULER] [SALIENCY] [MODEL]"}
LR=${2:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <DATA_PATH> [OPD_PROMPT_PATH] [MASK_INTERVAL] [LR_SCHEDULER] [SALIENCY] [MODEL]"}
KL_THRESHOLD=${3:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <DATA_PATH> [OPD_PROMPT_PATH] [MASK_INTERVAL] [LR_SCHEDULER] [SALIENCY] [MODEL]"}
DATA_PATH=${4:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <DATA_PATH> [OPD_PROMPT_PATH] [MASK_INTERVAL] [LR_SCHEDULER] [SALIENCY] [MODEL]"}
REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
OPD_PROMPT_PATH=${5:-${REPO_ROOT}/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl}
MASK_INTERVAL=${6:-32}
LR_SCHEDULER=${7:-cosine}
SALIENCY=${8:-fisher}
MODEL=${9:-Qwen/Qwen3-4B}

# wandb project follows the model size (reasoning_qwen3_4b / reasoning_qwen3_1.7b / ...),
# matching this session's existing per-size project convention.
MODEL_TAG=$(echo "$MODEL" | sed -E 's#.*Qwen3-##; s#B$#b#' | tr '[:upper:]' '[:lower:]')
WANDB_PROJECT="reasoning_qwen3_${MODEL_TAG}"

# Short, deduplicated dataset tag for the run name (strips the repetitive
# ot3_/fineweb_200k_/qwen3_ boilerplate shared by every dataset filename).
DATA_TAG=$(basename "$DATA_PATH" .jsonl | sed -E 's/ot3_?//g; s/fineweb_200k_?//g; s/qwen3_?//g; s/^_+//; s/_+$//; s/__+/_/g')

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

echo "=== TR-GMP NTP+KD+OPD (0.33/0.33/0.33) ${MODEL} s${SPARSITY} lr=${LR} kl=${KL_THRESHOLD} mask_interval=${MASK_INTERVAL} saliency=${SALIENCY} data=${DATA_TAG} ==="
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
    --gmp_saliency=${SALIENCY} \
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
    --wandb_project=${WANDB_PROJECT} \
    --seed=42 \
    --run_name_suffix="${SALIENCY}_${DATA_TAG}"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
