#!/bin/bash
#SBATCH --job-name=lighteval_only
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/lighteval_only_%j.out
#SBATCH --exclude=n3,n42,n51,n52,n54,n55,n58,n60,n76,n77,n80
exec 2>&1

# lighteval-only eval — resumes existing wandb run
# Usage: sbatch slurm_lighteval_only.sh <MODEL_PATH> <WANDB_RUN_ID>

MODEL_PATH=${1:?"Usage: sbatch slurm_lighteval_only.sh <MODEL_PATH> <WANDB_RUN_ID>"}
WANDB_RUN_ID=${2:?"Usage: sbatch slurm_lighteval_only.sh <MODEL_PATH> <WANDB_RUN_ID>"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

mkdir -p /home1/doyoonkim/projects/elsa/logs

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TMPDIR=/tmp

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL=$MODEL_PATH"
echo "WANDB_RUN_ID=$WANDB_RUN_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project reasoning_qwen3_1.7b \
    --wandb_run_id "$WANDB_RUN_ID" \
    --method auto \
    --skip_ppl \
    --skip_zeroshot \
    --gpu_util 0.85 \
    --tp_size 1

echo "##### END #####"
