#!/bin/bash
#SBATCH --job-name=eval_dense_8b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_dense_qwen3_8b_%j.out
#SBATCH --exclude=n3,n42,n60,n76,n77,n80
exec 2>&1

# Find Qwen3-8B snapshot path dynamically
MODEL=$(ls -d /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/  2>/dev/null | head -1)
MODEL="${MODEL%/}"
if [ -z "$MODEL" ] || [ ! -f "$MODEL/config.json" ]; then
    echo "ERROR: Qwen3-8B model not found in HF cache. Download first." >&2
    exit 1
fi

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "${LOCAL_JOB_BASE}/eval_out" "${LOCAL_JOB_BASE}/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
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

echo "=== eval dense Qwen3-8B (1 GPU) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/eval_full.py \
    --model_path "${MODEL}" \
    --wandb_project reasoning_qwen3_4b \
    --run_name "dense_qwen3_8b" \
    --method dense \
    --sparsity 0.0 \
    --gpu_util 0.85 \
    --tp_size 1 \
    --out_base "${LOCAL_JOB_BASE}/eval_out"

echo "##### END #####"
