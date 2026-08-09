#!/bin/bash
#SBATCH --job-name=prebuild_cache
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/prebuild_cache_%j.out
exec 2>&1

export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATA_PATH=${1:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl}
MAX_LEN=${2:-2048}

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  DATA_PATH=$DATA_PATH  MAX_LEN=$MAX_LEN"
/home1/doyoonkim/miniconda3/envs/rac/bin/python /home1/doyoonkim/projects/elsa/scripts/prebuild_mixed_cot_cache.py "$DATA_PATH" "$MAX_LEN"
echo "##### END #####"
