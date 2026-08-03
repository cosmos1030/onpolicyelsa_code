#!/bin/bash
#SBATCH --job-name=elsa_plain_8b_h200
#SBATCH --partition=H200
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n76,n77,n80,n87,n91,n61,n64
#SBATCH --output=/local-data/user-data/%u/elsa_plain_8b_h200_%j/slurm_%j.out
exec 2>&1

# Plain ELSA NTP-ADMM, Qwen3-8B, 2x H200 FSDP, OT80/FW20 rerun
# H200 alternative to the 4xA100-80GB variant — A100-80GB queue is congested
# (93 jobs) while H200 has spare capacity. grad_accum doubled (8 instead of 4)
# to keep global batch=16 with half the world_size.
# n87 excluded: known no-internet node (infra_broken_nodes memory) — only
# n88 is used, so this only actually runs if n88 has free GPUs.
#
# Usage: sbatch slurm_elsa_plain_qwen3_8b_h200.sh <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE]
#   sparsity=0.5 → lr=5e-5, lmda=1e-3
#   sparsity=0.6 → lr=5e-5, lmda=5e-3
#   sparsity=0.7 → lr=1e-4, lmda=5e-3
#   2:4 (sparsity=0.5, sparsity_type=2:4) → lr=5e-5, lmda=5e-3

SPARSITY=${1:?"Usage: sbatch slurm_elsa_plain_qwen3_8b_h200.sh <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE]"}
LR=${2:?"Usage: sbatch slurm_elsa_plain_qwen3_8b_h200.sh <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE]"}
LMDA=${3:?"Usage: sbatch slurm_elsa_plain_qwen3_8b_h200.sh <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE]"}
SPARSITY_TYPE=${4:-unstructured}

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL=$(ls -d /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ 2>/dev/null | head -1)
MODEL="${MODEL%/}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL/config.json" ]; then
    echo "ERROR: Qwen3-8B not found in HF cache" >&2
    exit 1
fi
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

LOCAL_JOB_BASE="/local-data/user-data/${USER}/elsa_plain_8b_h200_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

# Node access is revoked once the job ends, so /local-data logs become
# permanently unreachable if the job crashes — copy the log to NFS once on
# exit (single small write, not continuous I/O) so post-mortem is possible.
DEBUG_LOG_COPY="/home1/doyoonkim/projects/elsa/logs/elsa_plain_8b_h200_${SLURM_JOB_ID}_last.out"
mkdir -p /home1/doyoonkim/projects/elsa/logs
copy_log_on_exit() { cp "$LOCAL_JOB_BASE/slurm_${SLURM_JOB_ID}.out" "$DEBUG_LOG_COPY" 2>/dev/null || true; }
trap copy_log_on_exit EXIT

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export WANDB_RUN_ID_OUTPUT="/home1/doyoonkim/projects/elsa/logs/handoff_${SLURM_JOB_ID}_wandb_run_id.txt"
export MODEL_PATH_OUTPUT="/home1/doyoonkim/projects/elsa/logs/handoff_${SLURM_JOB_ID}_model_path.txt"
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

echo "=== Plain ELSA NTP-ADMM Qwen3-8B s${SPARSITY_PCT} (${SPARSITY_TYPE}), lr=${LR}, lmda=${LMDA}, 2xH200 FSDP (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=2 --master_port=${MASTER_PORT} main.py \
    --model="$MODEL" \
    --data_path="$DATA_PATH" \
    --dataset=mixed_cot \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=${SPARSITY_TYPE} \
    --steps=4096 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=8 \
    --admm_use_fsdp=true \
    --admm_lmda=${LMDA} \
    --admm_lmda_schedule_mode=constant \
    --lr=${LR} \
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
    --save_model=true \
    --admm_save_path=/home1/doyoonkim/projects/elsa/models \
    --eval_math500=false \
    --eval_zero_shot=false \
    --eval_full_bench=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_8b \
    --seed=42 \
    --push_to_hub=true

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
