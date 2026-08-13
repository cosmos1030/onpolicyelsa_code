#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opkd_24_4b
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/tr_ntpkd_opkd_24_4b_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) for Qwen3-4B, 2:4 semi-structured
# sparsity, single H200 (no FSDP) -- log_cluster port of the other server's
# slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b.sh (same recipe, already validated
# there off the 1.7B 2:4 canary, job 718464). FSDP was tried at 4B there too
# and crashed in loss.backward() with a tensor-shape mismatch: the
# structured-L1 (2:4 grouping) code assumes full unsharded 2D weight
# tensors, but FSDP hands each rank a flat 1D shard -- fundamentally
# incompatible with FSDP as currently written, so this stays single-GPU.
# 4B unstructured TR-GMP already runs fine single-GPU on this cluster (only
# 8B needed FSDP, see slurm_gmp_tr_ntpkd_opd_qwen3_8b_fsdp2gpu.sh), and the
# 2:4 L1 regularizer is a cheap elementwise op, so headroom should be fine.
#
# gmp_l1_lambda is the "lasso" term: an L1 penalty pulling weights toward
# the eventual 2:4 grouping before the mask is finalized -- see the recent
# "drop lasso after mask freeze" fix (2:4 structured-L1 via forward
# pre-hooks, exact N:M/0-tolerance TR-reached check) pulled in from the
# other server's session -- it's a pre-shrink aid during TR-GMP's mask
# search, not a permanent loss term once sparsity is reached.
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b.sh <SPARSITY> <LR> <KL_THRESHOLD> \
#          [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH] [WANDB_PROJECT]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b.sh 0.5 1e-4 0.02

SPARSITY=${1:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH] [WANDB_PROJECT]"}
LR=${2:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH] [WANDB_PROJECT]"}
KL_THRESHOLD=${3:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH] [WANDB_PROJECT]"}
MASK_INTERVAL=${4:-32}
L1_LAMBDA=${5:-0.0001}
REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
DATA_PATH=${6:-${REPO_ROOT}/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
WANDB_PROJECT=${7:-reasoning_qwen3_4b_nostrip8192}
MODEL="Qwen/Qwen3-4B"
SEQLEN=8192
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
# NOT expandable_segments:True -- OPKD's vLLM engine runs with
# enable_sleep_mode=True (single-GPU path, see lib/gmp_trainer.py
# _opkd_vllm_wake/sleep), whose CuMemAllocator hard-asserts expandable_segments
# is unset at load_model() time. Use max_split_size_mb instead -- a different
# fragmentation mitigation the CuMemAllocator assertion doesn't check for (it
# only greps for the literal string "expandable_segments:True") -- leaving
# fragmentation completely unmitigated caused a real OOM after ~760 steps on
# the other server's 1.7B single-GPU 2:4 canary (job 720073).
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "=== TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) 2:4 ${MODEL} s${SPARSITY} lr=${LR} kl=${KL_THRESHOLD} mi=${MASK_INTERVAL} l1=${L1_LAMBDA} (single H200) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
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
    --sparsity_type=2:4 \
    --gmp_l1_lambda=${L1_LAMBDA} \
    --do_gmp=true \
    --steps=2048 \
    --gmp_post_target_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_use_fsdp=false \
    --gmp_save_path="${REPO_ROOT}/models" \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_profile=quick \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --run_name_suffix="24_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
