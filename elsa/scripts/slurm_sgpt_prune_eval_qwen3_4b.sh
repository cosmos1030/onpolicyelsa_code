#!/bin/bash
#SBATCH --job-name=sgpt_qwen3_4b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n51,n52,n54,n55,n58,n60,n76,n80
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/sgpt_qwen3_4b_%j.out
exec 2>&1

# SparseGPT prune + full eval for Qwen3-4B → reasoning_qwen3_4b
# Usage: sbatch slurm_sgpt_prune_eval_qwen3_4b.sh <SPARSITY> [NSAMPLES=128]

SPARSITY=${1:?"Usage: sbatch slurm_sgpt_prune_eval_qwen3_4b.sh <SPARSITY> [NSAMPLES]"}
NSAMPLES=${2:-128}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
SAVE_PATH="/home1/doyoonkim/projects/elsa/models/qwen3_4b_sgpt_s${SPARSITY_PCT}pct_n${NSAMPLES}"

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
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "=== SparseGPT Qwen3-4B (s${SPARSITY_PCT}%) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "SPARSITY=$SPARSITY  NSAMPLES=$NSAMPLES"
echo "SAVE_PATH=$SAVE_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/RAC/open-r1-main

$PYTHON src/open_r1/prune_and_eval.py \
    --model_path "$MODEL" \
    --sparsity "$SPARSITY" \
    --nsamples "$NSAMPLES" \
    --seqlen 2048 \
    --save_path "$SAVE_PATH" \
    --wandb_project reasoning_qwen3_4b \
    --wandb_name "sgpt_s${SPARSITY_PCT}"

echo "##### END #####"
