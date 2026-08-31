#!/bin/bash
#SBATCH --job-name=sparsellm_1.7b
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --output=/home1/doyoonkim/projects/SparseLLM/logs/sparsellm_s%a_%j.out
#SBATCH --exclude=n3,n42,n51,n52,n54,n55,n58,n60,n76,n77,n80
exec 2>&1

# SparseLLM pruning + full eval for Qwen3-1.7B on RTX3090
# Usage: sbatch slurm_sparsellm_3090_1.7b.sh <SPARSITY>

SPARSITY=${1:?"Usage: sbatch slurm_sparsellm_3090_1.7b.sh <SPARSITY>"}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"  # matched to ALPS calibration set (was ot3_fineweb_200k_qwen3.jsonl, a different/uncleaned file)
SAVE_BASE="/home1/doyoonkim/projects/elsa/models"
SAVED_MODEL="${SAVE_BASE}/qwen3_1.7b_sparsellm_s${SPARSITY_PCT}pct"

mkdir -p /home1/doyoonkim/projects/SparseLLM/logs

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

echo "=== SparseLLM Qwen3-1.7B s${SPARSITY_PCT}% ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/SparseLLM

$PYTHON qwen3_main.py \
    --model "$MODEL" \
    --data_path "$DATA" \
    --nsamples 128 \
    --seqlen 2048 \
    --sparsity ${SPARSITY} \
    --seed 42 \
    --save "$SAVED_MODEL" \
    --eval_full \
    --wandb_project reasoning_qwen3_1.7b \
    --run_name "sparsellm_s${SPARSITY_PCT}pct" \
    --gpu_util 0.85 \
    --tp_size 1 \
    --profile quick

EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE ==="
echo "##### END #####"
