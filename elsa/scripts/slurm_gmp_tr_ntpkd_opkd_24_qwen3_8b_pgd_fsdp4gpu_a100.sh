#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opkd_24_8b_pgd_a100
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n79,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_ntpkd_opkd_24_8b_pgd_a100_%j.out
exec 2>&1

# 8B analogue of slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_pgd_fsdp4gpu_v2.sh: full
# TR-GMP (not fixed-mask) NTP+KD+OPKD(0.33/0.33/0.33), 2:4 semi-structured,
# PGD reprojection every step, structured-L1, Qwen3-8B, dense start (not the
# ALPS checkpoint -- this is the TR-GMP-from-scratch N24 comparison point,
# same as the 4B/1.7B 733812/733813 runs).
#
# 4x A100-80GB total, vLLM sharing training rank 0's GPU
# (gmp_opkd_vllm_gpu_index=0, TP=1) instead of a 5th dedicated GPU. NOTE:
# the exact 4B analogue of this config (job 736110) OOM'd on GPU0 during an
# OPKD rollout burst (66.46GB in use, only 1.29GB free for a 2.32GB alloc)
# -- 8B being ~2x the params makes this MORE likely to OOM, not less. Kept
# at 4 GPUs total per explicit instruction rather than the isolated-vLLM
# 5-GPU fallback (slurm_gmp_tr_ntpkd_opkd_24_qwen3_8b_pgd_fsdp4gpu_h200.sh)
# that's actually proven safe for this combination. If this OOMs, that H200
# script is the known-working fallback.
#
# global batch = gmp_batch_size(1) * gmp_grad_accum(2) * world_size(4) = 8,
# matching the 4B/1.7B PGD scripts' global batch for a budget-matched
# comparison.
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_8b_pgd_fsdp4gpu_a100.sh <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_8b_pgd_fsdp4gpu_a100.sh 0.5 1e-4 0.02

SPARSITY=${1:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]"}
LR=${2:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]"}
KL_THRESHOLD=${3:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]"}
MASK_INTERVAL=${4:-32}
L1_LAMBDA=${5:-0.0001}
DATA_PATH=${6:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL=$(ls -d /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ 2>/dev/null | head -1)
MODEL="${MODEL%/}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL/config.json" ]; then
    echo "ERROR: Qwen3-8B not found in HF cache" >&2
    exit 1
fi
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"
SEQLEN=8192

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
# expandable_segments is incompatible with vLLM's CuMemAllocator (OPKD
# sleep-mode engine hard-asserts on it) -- see infra_vllm_sleepmode_expandable_segments memory.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) 2:4+PGD+lasso Qwen3-8B s${SPARSITY_PCT} lr=${LR} kl=${KL_THRESHOLD} mi=${MASK_INTERVAL} l1=${L1_LAMBDA}, 4xA100-80GB FSDP (vLLM shares GPU0), global_batch=8 (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$MODEL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=4 --master_port=${MASTER_PORT} main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=2:4 \
    --gmp_l1_lambda=${L1_LAMBDA} \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --steps=2048 \
    --gmp_post_target_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=2 \
    --lr=${LR} \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=true \
    --gmp_kl_chunk_size=1024 \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.22 \
    --gmp_opkd_vllm_gpu_index=0 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_pgd=true \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_8b_nostrip8192 \
    --run_name_suffix="24_pgd_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}_h200x4" \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
