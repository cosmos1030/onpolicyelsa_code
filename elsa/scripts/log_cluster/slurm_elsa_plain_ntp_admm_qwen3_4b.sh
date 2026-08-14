#!/bin/bash
#SBATCH --job-name=elsa_plain_ntp_4b
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/elsa_plain_ntp_4b_%j.out
exec 2>&1

# Plain ELSA NTP-ADMM baseline, Qwen3-4B, single H200 (no FSDP -- NTP-only
# ADMM has no KD teacher/vLLM, so it's much lighter than the TR-GMP
# NTP+KD+OPKD recipe that already fits single-GPU at 4B), matched to the
# CURRENT TR-GMP nostrip8192 sweep's settings for a fair comparison:
# data=nostrip8192, seqlen=8192, steps=2048, global batch=8 (batch=1 x
# grad_accum=8), lr_scheduler=cosine + 256-step warmup (TR-GMP's schedule,
# not the old elsa_plain scripts' constant_with_warmup).
#
# admm_lmda_schedule_mode=cosine (lmda ramps up smoothly, not held constant
# -- this is "plain ELSA-like" behavior) is fixed; admm_lmda ITSELF is what
# the sweep varies. Reference values (rerun_ot80fw20/slurm_elsa_plain_qwen3_4b.sh,
# old OT80/FW20 data): s50/lr5e-5->1e-3, s60/lr5e-5->5e-3, s70/lr1e-4->5e-3,
# 2:4/lr5e-5->5e-3. SPARSITY_TYPE=2:4 needs prune_n=2/prune_m=4, which
# lib/prune.py's globalprune_admm already threads through to
# AdmmTrainingArguments (prune_n/prune_m fields) -- no code changes needed,
# just pass sparsity_type=2:4 (sparsity_ratio is ignored by the N:M path but
# still required positionally; conventionally passed as 0.5 to match 2:4's
# fixed 50%).
#
# Usage: sbatch slurm_elsa_plain_ntp_admm_qwen3_4b.sh <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE] [DATA_PATH] [WANDB_PROJECT]
# e.g.: sbatch slurm_elsa_plain_ntp_admm_qwen3_4b.sh 0.5 5e-5 1e-3
#       sbatch slurm_elsa_plain_ntp_admm_qwen3_4b.sh 0.5 5e-5 5e-3 2:4

SPARSITY=${1:?"Usage: <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE] [DATA_PATH] [WANDB_PROJECT]"}
LR=${2:?"Usage: <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE] [DATA_PATH] [WANDB_PROJECT]"}
LMDA=${3:?"Usage: <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE] [DATA_PATH] [WANDB_PROJECT]"}
SPARSITY_TYPE=${4:-unstructured}
REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
DATA_PATH=${5:-${REPO_ROOT}/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
WANDB_PROJECT=${6:-reasoning_qwen3_4b_nostrip8192}
MODEL="Qwen/Qwen3-4B"
SEQLEN=8192
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

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
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "=== Plain ELSA NTP-ADMM ${MODEL} s${SPARSITY_PCT} (${SPARSITY_TYPE}) lr=${LR} lmda=${LMDA} (cosine schedule) data=${DATA_TAG} (single H200) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

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
    --sparsity_type=${SPARSITY_TYPE} \
    --steps=2048 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=8 \
    --admm_use_fsdp=false \
    --admm_lmda=${LMDA} \
    --admm_lmda_schedule_mode=cosine \
    --lr=${LR} \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --seqlen=${SEQLEN} \
    --admm_base_optimizer=adamw \
    --admm_beta1=0.9 \
    --admm_beta2=0.999 \
    --admm_projection_mode=momentum \
    --admm_interval=32 \
    --admm_precision=bf16 \
    --admm_dual_dtype=fp32 \
    --admm_split_dtype=fp32 \
    --save_model=true \
    --admm_save_path="${REPO_ROOT}/models" \
    --eval_math500=false \
    --eval_zero_shot=true \
    --eval_full_bench=true \
    --eval_profile=quick \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --seed=42 \
    --push_to_hub=true

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
