#!/bin/bash
#SBATCH --job-name=eval_dense_qwen3_4b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n51,n52,n54,n55,n58,n60,n76,n80
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_dense_qwen3_4b_%j.out
exec 2>&1

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
OUT_BASE="$LOCAL_JOB_BASE/eval_dense"

echo "=== eval_full dense Qwen3-4B ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL" \
    --wandb_project reasoning_qwen3_4b \
    --run_name dense \
    --method dense \
    --sparsity 0.0 \
    --gpu_util 0.85 \
    --out_base "$OUT_BASE"

echo "##### END #####"
