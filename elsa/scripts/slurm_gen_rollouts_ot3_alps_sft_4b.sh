#!/bin/bash
#SBATCH --job-name=gen_ot3_alps_sft_4b
#SBATCH --partition=RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/gen_ot3_alps_sft_4b_%j.out
exec 2>&1

# Generate 20k rollouts (2048 max tokens) from the ALPS one-shot + sparse SFT
# (NTP, fixed mask) Qwen3-4B s70pct checkpoint (job 692163) on
# OpenThoughts3-1.2M prompts, then push the dataset to HuggingFace.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export TMPDIR=/tmp

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa/scripts
$PYTHON gen_rollouts_ot3_alps_sft_4b.py
EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
