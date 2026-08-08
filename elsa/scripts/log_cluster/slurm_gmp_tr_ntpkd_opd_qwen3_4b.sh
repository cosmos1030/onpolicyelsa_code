#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opd_4b
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/tr_ntpkd_opd_4b_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPD (0.33/0.33/0.33), Qwen3-4B, dense -> gradual Fisher-saliency
# pruning to target sparsity via trust-region mask growth, OT80/FW20.
#
# Same config as the KD+OPD-only (no NTP) 4B reference runs 696316 (s50) /
# 697311 (s60) / 697312 (s70) -- confirmed via their wandb configs (wbid
# plqvpkqo/q2xlvti9/lrwf1ytg) -- except:
#   - gmp_ntp_lambda: 0.0 (via gmp_kd_only=true) -> 0.33 (NTP now included)
#   - lr_scheduler: constant_with_warmup -> cosine (the thing being tested:
#     702463's investigation found the unified `lr_scheduler` flag defaults
#     to constant_with_warmup since commit 013e9df, silently dropping the
#     intended cosine decay unless passed explicitly)
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b.sh <SPARSITY> [LR_SCHEDULER]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b.sh 0.5
#       sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b.sh 0.6
#       sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b.sh 0.7

SPARSITY=${1:?"Usage: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b.sh <SPARSITY> [LR_SCHEDULER] [MASK_INTERVAL]"}
LR_SCHEDULER=${2:-cosine}
MASK_INTERVAL=${3:-32}

MODEL="Qwen/Qwen3-4B"
REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
DATA_PATH="${REPO_ROOT}/data/ot3_fineweb_200k_qwen3_train.jsonl"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "=== TR-GMP NTP+KD+OPD (0.33/0.33/0.33) Qwen3-4B s${SPARSITY} lr_scheduler=${LR_SCHEDULER} (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$MODEL"
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
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=1e-4 \
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
    --gmp_tr_kl_threshold=0.01 \
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
