#!/bin/bash
#SBATCH --job-name=elsa_cubic_lw_ext_3090
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/elsa_cubic_lw_ext_3090_%j/slurm_%j.out
exec 2>&1

# Same as slurm_elsa_cubic_layerwise_ntp_s70_lr1e4_1.7b_fsdp4_3090.sh (cubic,
# layerwise, s70, lr1e-4, interval=32, admm_cubic_steps=1024 ramp) but
# admm_steps=3072 instead of 2048 — i.e. cubic still reaches final sparsity at
# step 1024, but then gets 2048 MORE steps at the fixed final mask, matching
# plain's full post-target adaptation budget (plain is fixed from step 32,
# effectively ~2048 steps of adaptation). NOT an apples-to-apples total-budget
# comparison (this run gets 3072 steps total vs plain's 2048) — this is
# specifically to test whether cubic's WikiText2 PPL gap vs plain closes once
# it gets equal TIME-AT-FINAL-STRUCTURE, not equal total compute.
# Usage: sbatch slurm_elsa_cubic_layerwise_extended_ntp_s70_lr1e4_1.7b_fsdp4_3090.sh <FINAL_LMDA>

FINAL_LMDA=${1:-0.005}

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/elsa_cubic_lw_ext_3090_${SLURM_JOB_ID}"
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

echo "=== Cubic-z LAYERWISE EXTENDED (3072 steps, ks=1024) NTP-ADMM Qwen3-1.7B s70, 4xRTX3090 FSDP, final_lmda=${FINAL_LMDA} ==="
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
    --admm_steps=3072 \
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
    --admm_z_schedule_mode=cubic \
    --admm_cubic_steps=1024 \
    --admm_z_layerwise=true \
    --save_model=false \
    --eval_math500=false \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42 \
    --push_to_hub=false

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
