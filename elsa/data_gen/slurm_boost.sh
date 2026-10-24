#!/bin/bash
#SBATCH --job-name=boost_math
#SBATCH --partition=A6000,L40S,RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=n3,n42,n43,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n84,n87,n90,n91,n92
#SBATCH --output=/home1/doyoonkim/projects/elsa/data_gen/logs/%x_%j.out
exec 2>&1
# Usage: sbatch slurm_boost.sh <START> <END> <OUT_NAME> [extra boost_batched.py args]
START=${1:?start}; END=${2:?end}; NAME=${3:?out name}; shift 3
export HF_HOME=/home1/doyoonkim/.cache/huggingface TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_HOST_IP=127.0.0.1
cd /home1/doyoonkim/projects/elsa/data_gen
/home1/doyoonkim/miniconda3/envs/rac/bin/python -u boost_batched.py \
    --model /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --data ${BOOST_DATA:-/home1/doyoonkim/projects/finetuning-with-sampling/math_data/train.parquet} \
    --start $START --end $END --out out/$NAME.jsonl "$@"
