#!/bin/bash
#SBATCH --job-name=elsa_cubic_1.7b
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91,n61,n64,n31
#SBATCH --output=/local-data/user-data/%u/elsa_cubic_1.7b_%j/slurm_%j.out
exec 2>&1

# Cubic z-projection NTP-ADMM, Qwen3-1.7B, 4x RTX3090 FSDP, OT80/FW20 rerun.
# Same config as slurm_elsa_plain_qwen3_1.7b.sh (matches wandb run olrirliu:
# s50, lr=1e-4, lmda=1e-3, steps=4096, global batch=16) with the cubic
# z-projection schedule added on top: sparsity ramps 0→final over the first
# admm_cubic_steps steps (Boza et al. cubic schedule), then holds at final
# sparsity for the remaining steps (lib/trainer.py:_compute_cubic_z).
#
# Usage: sbatch slurm_elsa_cubic_qwen3_1.7b.sh <SPARSITY> <LMDA> <CUBIC_STEPS> [SPARSITY_TYPE]
# e.g.: sbatch slurm_elsa_cubic_qwen3_1.7b.sh 0.5 1e-3 2048

SPARSITY=${1:?"Usage: sbatch slurm_elsa_cubic_qwen3_1.7b.sh <SPARSITY> <LMDA> <CUBIC_STEPS> [SPARSITY_TYPE]"}
LMDA=${2:?"Usage: sbatch slurm_elsa_cubic_qwen3_1.7b.sh <SPARSITY> <LMDA> <CUBIC_STEPS> [SPARSITY_TYPE]"}
CUBIC_STEPS=${3:?"Usage: sbatch slurm_elsa_cubic_qwen3_1.7b.sh <SPARSITY> <LMDA> <CUBIC_STEPS> [SPARSITY_TYPE]"}
SPARSITY_TYPE=${4:-unstructured}

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

LOCAL_JOB_BASE="/local-data/user-data/${USER}/elsa_cubic_1.7b_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

# Node access is revoked once the job ends, so /local-data logs become
# permanently unreachable if the job crashes — copy the log to NFS once on
# exit (single small write, not continuous I/O) so post-mortem is possible.
DEBUG_LOG_COPY="/home1/doyoonkim/projects/elsa/logs/elsa_cubic_1.7b_${SLURM_JOB_ID}_last.out"
mkdir -p /home1/doyoonkim/projects/elsa/logs
copy_log_on_exit() { cp "$LOCAL_JOB_BASE/slurm_${SLURM_JOB_ID}.out" "$DEBUG_LOG_COPY" 2>/dev/null || true; }
trap copy_log_on_exit EXIT

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

echo "=== Cubic-z NTP-ADMM Qwen3-1.7B s${SPARSITY_PCT} (${SPARSITY_TYPE}), lmda=${LMDA}, cubic_steps=${CUBIC_STEPS}, 4xRTX3090 FSDP (OT80/FW20) ==="
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
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=${SPARSITY_TYPE} \
    --steps=4096 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=4 \
    --admm_use_fsdp=true \
    --admm_lmda=${LMDA} \
    --admm_lmda_schedule_mode=constant \
    --lr=1e-4 \
    --lr_scheduler=constant_with_warmup \
    --lr_warmup_steps=256 \
    --seqlen=2048 \
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
    --admm_cubic_steps=${CUBIC_STEPS} \
    --save_model=true \
    --admm_save_path=/home1/doyoonkim/projects/elsa/models \
    --eval_math500=false \
    --eval_zero_shot=false \
    --eval_full_bench=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42 \
    --push_to_hub=true

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
