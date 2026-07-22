#!/bin/bash
#SBATCH --job-name=eval_ms
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_ms_%j.out
#SBATCH --exclude=n3,n60,n76,n80

# Usage: sbatch slurm_eval_milestone.sh <MODEL_PATH> <RUN_NAME> <SPARSITY>
# Example: sbatch slurm_eval_milestone.sh /home1/.../models/ckpt tr_gmp_kl001_s70 0.70

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_milestone.sh <MODEL_PATH> <RUN_NAME> <SPARSITY> [extra_flags...]"}
RUN_NAME=${2:?"Usage: sbatch slurm_eval_milestone.sh <MODEL_PATH> <RUN_NAME> <SPARSITY> [extra_flags...]"}
SPARSITY=${3:-0.0}
EXTRA_FLAGS="${@:4}"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "${LOCAL_JOB_BASE}/eval_out"
mkdir -p /home1/doyoonkim/projects/elsa/logs
exec 2>&1

echo "=== eval_milestone: ${RUN_NAME} sp=${SPARSITY} ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL=${MODEL_PATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
# WANDB_API_KEY: SLURM이 전달한 값 우선, 없으면 .bashrc에서 읽기
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY=$(grep "^export WANDB_API_KEY=" ~/.bashrc 2>/dev/null | cut -d= -f2-)
fi
export WANDB_DIR="${LOCAL_JOB_BASE}/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
mkdir -p "${LOCAL_JOB_BASE}/wandb"
export TMPDIR=/tmp
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_CACHE="/home1/doyoonkim/.cache/huggingface/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

cd /home1/doyoonkim/projects/elsa

/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/eval_full.py \
    --model_path "${MODEL_PATH}" \
    --wandb_project reasoning_qwen3_4b \
    --run_name "${RUN_NAME}" \
    --method gmp_tr \
    --sparsity "${SPARSITY}" \
    --gpu_util 0.85 \
    --out_base "${LOCAL_JOB_BASE}/eval_out" \
    ${EXTRA_FLAGS}

echo "##### END #####"
