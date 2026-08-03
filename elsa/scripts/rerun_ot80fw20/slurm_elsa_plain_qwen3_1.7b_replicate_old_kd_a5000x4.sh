#!/bin/bash
#SBATCH --job-name=elsa_plain_1.7b_old_kd
#SBATCH --partition=A5000
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=150G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/elsa_plain_1.7b_old_kd_%j/slurm_%j.out
exec 2>&1

# Same as slurm_elsa_plain_qwen3_1.7b_replicate_old.sh (steps=2048, global
# batch=8, 4xA5000, lr_scheduler=linear/no-warmup, admm_lmda_schedule_mode
# =cosine, saliency=momentum) but with a dense-teacher KD loss added on top
# of NTP: do_kd_admm=true loads a fresh copy of the original (unpruned)
# model as teacher and mixes kd_lambda*KD + kd_ntp_lambda*NTP — set to
# 0.5/0.5 here. Teacher forward pass only (kd_use_vllm stays False, no
# on-policy generation), same input sequence as student, so no extra
# distributed/vLLM risk beyond a second ~1.7B model's memory footprint
# (comfortable on a single 80GB card).
#
# Usage: sbatch slurm_elsa_plain_qwen3_1.7b_replicate_old_kd.sh <SPARSITY> <LMDA> [SPARSITY_TYPE]
# e.g.: sbatch slurm_elsa_plain_qwen3_1.7b_replicate_old_kd.sh 0.5 1e-3

SPARSITY=${1:?"Usage: sbatch slurm_elsa_plain_qwen3_1.7b_replicate_old_kd.sh <SPARSITY> <LMDA> [SPARSITY_TYPE]"}
LMDA=${2:?"Usage: sbatch slurm_elsa_plain_qwen3_1.7b_replicate_old_kd.sh <SPARSITY> <LMDA> [SPARSITY_TYPE]"}
SPARSITY_TYPE=${3:-unstructured}

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

LOCAL_JOB_BASE="/local-data/user-data/${USER}/elsa_plain_1.7b_old_kd_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

DEBUG_LOG_COPY="/home1/doyoonkim/projects/elsa/logs/elsa_plain_1.7b_old_kd_${SLURM_JOB_ID}_last.out"
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

echo "=== ELSA plain+KD (replicate ef2zr9qy schedule) Qwen3-1.7B s${SPARSITY_PCT} (${SPARSITY_TYPE}), lmda=${LMDA}, kd=0.5/ntp=0.5, 4xA5000 (OT80/FW20) ==="
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
    --steps=2048 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=2 \
    --admm_use_fsdp=true \
    --admm_lmda=${LMDA} \
    --admm_lmda_schedule_mode=cosine \
    --lr=1e-4 \
    --lr_scheduler=linear \
    --lr_warmup_steps=0 \
    --seqlen=2048 \
    --admm_base_optimizer=adamw \
    --admm_beta1=0.9 \
    --admm_beta2=0.999 \
    --admm_projection_mode=momentum \
    --admm_interval=32 \
    --admm_precision=bf16 \
    --admm_dual_dtype=fp32 \
    --admm_split_dtype=fp32 \
    --do_kd_admm=true \
    --kd_lambda=0.5 \
    --kd_ntp_lambda=0.5 \
    --kd_use_vllm=false \
    --save_model=true \
    --admm_save_path=/home1/doyoonkim/projects/elsa/models \
    --eval_math500=false \
    --eval_zero_shot=true \
    --eval_full_bench=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42 \
    --push_to_hub=true

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
