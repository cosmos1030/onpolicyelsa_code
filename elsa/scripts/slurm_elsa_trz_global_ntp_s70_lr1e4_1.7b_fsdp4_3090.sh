#!/bin/bash
#SBATCH --job-name=elsa_trz_global_fsdp4
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/elsa_trz_global_fsdp4_%j/slurm_%j.out
exec 2>&1

# Trust-region (KL-gated, adaptive) z-projection NTP-ADMM, Qwen3-1.7B, 4x RTX3090
# FSDP, GLOBAL threshold (admm_z_layerwise=false — one saliency threshold pooled
# across all params, not per-tensor). Same base config (interval=32, lr=1e-4,
# constant lmda, momentum saliency) as the cubic-layerwise comparison runs, so
# results are directly comparable.
# Includes the just-added admm_tr_gate_at_target fix: once sparsity reaches the
# final target, further mask reselection is still KL-gated (swap/freeze) instead
# of reselecting unconditionally every interval.
# grad_accum=2 so global batch stays 1*2*4=8, matching every other comparison run.
# Usage: sbatch slurm_elsa_trz_global_ntp_s70_lr1e4_1.7b_fsdp4_3090.sh <KL_THRESHOLD> <FINAL_LMDA>

KL_THRESHOLD=${1:-0.1}
FINAL_LMDA=${2:-0.005}

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/elsa_trz_global_fsdp4_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/tmp
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== TR-z GLOBAL NTP-ADMM Qwen3-1.7B s70, 4xRTX3090 FSDP, kl=${KL_THRESHOLD}, final_lmda=${FINAL_LMDA} ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=4 --master_port=${MASTER_PORT} main.py \
    --model="$MODEL" \
    --data_path="$DATA_PATH" \
    --dataset=mixed_cot \
    --sparsity_ratio=0.7 \
    --admm_steps=2048 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=2 \
    --admm_use_fsdp=true \
    --admm_lmda=${FINAL_LMDA} \
    --admm_init_lmda=0 \
    --admm_final_lmda=${FINAL_LMDA} \
    --admm_lmda_schedule_mode=constant \
    --admm_lr=1e-4 \
    --admm_base_optimizer=adamw \
    --admm_beta1=0.9 \
    --admm_beta2=0.999 \
    --admm_projection_mode=momentum \
    --admm_interval=32 \
    --admm_precision=bf16 \
    --admm_dual_dtype=fp32 \
    --admm_split_dtype=fp32 \
    --admm_tr_z_proj=true \
    --admm_z_schedule_mode=trust_region \
    --admm_tr_kl_threshold=${KL_THRESHOLD} \
    --admm_tr_init_delta=0.05 \
    --admm_tr_delta_min=0.001 \
    --admm_tr_max_iters=8 \
    --admm_tr_gate_at_target=true \
    --admm_z_layerwise=false \
    --save_model=true \
    --admm_save_path=/home1/doyoonkim/projects/elsa/models \
    --eval_math500=false \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42 \
    --push_to_hub=true

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
