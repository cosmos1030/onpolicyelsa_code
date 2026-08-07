#!/bin/bash
#SBATCH --job-name=alps_sft_ntp_4b_fsdp2
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/alps_sft_ntp_4b_fsdp2_%j.out
exec 2>&1

# ALPS (one-shot pruned) -> fixed-mask NTP-ONLY recovery training, Qwen3-4B,
# OT80/FW20. Same recipe/budget as the 1.7B counterpart
# (slurm_alps_sft_ntp_qwen3_1.7b.sh: steps=2048, cosine-with-warmup,
# gmp_kd_lambda=0 so no teacher is loaded at all), just on 2xA100-80GB FSDP
# instead of single-GPU since 4B doesn't fit unsharded (see
# rerun_ot80fw20/slurm_alps_sparse_sft_qwen3_4b.sh's 4-GPU precedent, which
# used steps=4096 -- this run uses the shorter 2048-step budget to stay
# directly comparable to the 1.7B family instead).
#
# GradualMaskManager.init_from_weights() (gmp_fixed_mask=true) reads local
# per-rank shards under FSDP (use_orig_params=True) -- already proven correct
# by job 692163 (4-GPU ALPS NTP-only sparse SFT -> cosmos1030/alps-sparse-sft-
# qwen3-4b-s70pct).
#
# Usage: sbatch slurm_alps_sft_ntp_qwen3_4b_fsdp2gpu.sh <SPARSITY> [SPARSITY_TYPE] [LR_SCHEDULER]
# e.g.: sbatch slurm_alps_sft_ntp_qwen3_4b_fsdp2gpu.sh 0.5
#       sbatch slurm_alps_sft_ntp_qwen3_4b_fsdp2gpu.sh 0.5 2:4

SPARSITY=${1:?"Usage: sbatch slurm_alps_sft_ntp_qwen3_4b_fsdp2gpu.sh <SPARSITY> [SPARSITY_TYPE] [LR_SCHEDULER]"}
SPARSITY_TYPE=${2:-unstructured}
LR_SCHEDULER=${3:-cosine}

if [ "$SPARSITY_TYPE" = "2:4" ]; then
    ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_s24"
    SPARSITY_TAG="n24"
else
    SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
    ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_s${SPARSITY_PCT}pct"
    SPARSITY_TAG="s${SPARSITY_PCT}pct"
fi
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_train.jsonl"

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== ALPS -> NTP-only recovery training Qwen3-4B ${SPARSITY_TAG} (${SPARSITY_TYPE}) lr_scheduler=${LR_SCHEDULER}, 2xA100-80GB FSDP (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$ALPS_MODEL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=2 --master_port=${MASTER_PORT} main.py \
    --model="$ALPS_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=${SPARSITY_TYPE} \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --gmp_use_fsdp=true \
    --steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --lr=1e-4 \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --seqlen=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=1.0 \
    --gmp_kd_lambda=0.0 \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
