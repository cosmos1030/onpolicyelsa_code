#!/bin/bash
#SBATCH --job-name=ppl_only
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=/local-data/user-data/doyoonkim/logs/ppl_only_%j.out
#SBATCH --exclude=n3,n42,n51,n52,n54,n55,n58,n60,n76,n77,n80
exec 2>&1

# PPL-only eval — resumes existing wandb run
# Usage: sbatch slurm_ppl_only.sh <MODEL_PATH> <WANDB_RUN_ID>

MODEL_PATH=${1:?"Usage: sbatch slurm_ppl_only.sh <MODEL_PATH> <WANDB_RUN_ID>"}
WANDB_RUN_ID=${2:?"Usage: sbatch slurm_ppl_only.sh <MODEL_PATH> <WANDB_RUN_ID>"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

mkdir -p /local-data/user-data/doyoonkim/logs

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    --method elsa \
    --skip_lighteval

echo "##### END #####"

cp /local-data/user-data/doyoonkim/logs/ppl_only_${SLURM_JOB_ID}.out \
   /home1/doyoonkim/projects/elsa/logs/ppl_only_${SLURM_JOB_ID}.out 2>/dev/null || true
