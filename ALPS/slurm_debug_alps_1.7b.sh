#!/bin/bash
#SBATCH --job-name=alps_debug
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00
#SBATCH --output=/home1/doyoonkim/projects/ALPS/logs/debug_%j.out
exec 2>&1

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"

export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/ALPS

$PYTHON qwen3_alps.py \
    "$MODEL" \
    0.5 \
    --data_path "$DATA" \
    --nsamples 8 \
    --rho 300.0 \
    --seed 42

echo "##### END #####"
