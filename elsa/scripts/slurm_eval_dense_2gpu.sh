#!/bin/bash
#SBATCH --job-name=eval_dense_2gpu
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_dense_2gpu_%j.out
#SBATCH --exclude=n3,n60,n80
exec 2>&1

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "${LOCAL_JOB_BASE}/eval_out" "${LOCAL_JOB_BASE}/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY=$(grep "^export WANDB_API_KEY=" ~/.bashrc 2>/dev/null | cut -d= -f2-)
fi
export WANDB_DIR="${LOCAL_JOB_BASE}/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export TMPDIR=/tmp
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_CACHE="/home1/doyoonkim/.cache/huggingface/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "=== eval dense Qwen3-4B 2-GPU test ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/eval_full.py \
    --model_path "${MODEL}" \
    --wandb_project reasoning_qwen3_4b \
    --run_name "dense_qwen3_4b_2gpu_test" \
    --method dense \
    --sparsity 0.0 \
    --gpu_util 0.85 \
    --tp_size 2 \
    --out_base "${LOCAL_JOB_BASE}/eval_out"

echo "##### END #####"
