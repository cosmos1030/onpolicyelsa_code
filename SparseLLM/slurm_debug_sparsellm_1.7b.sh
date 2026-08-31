#!/bin/bash
#SBATCH --job-name=sparsellm_debug
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-01:00:00
#SBATCH --output=/home1/doyoonkim/projects/SparseLLM/logs/debug_%j.out
exec 2>&1

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"  # matched to ALPS calibration set (was ot3_fineweb_200k_qwen3.jsonl, a different/uncleaned file)

export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

mkdir -p /home1/doyoonkim/projects/SparseLLM/logs
cd /home1/doyoonkim/projects/SparseLLM

$PYTHON qwen3_main.py \
    --model "$MODEL" \
    --data_path "$DATA" \
    --nsamples 8 \
    --seqlen 2048 \
    --sparsity 0.5 \
    --seed 42

echo "##### END #####"
