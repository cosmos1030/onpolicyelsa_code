#!/bin/bash
#SBATCH --job-name=dbg_fmt_dense
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dbg_fmt_dense_%j.out

exec 2>&1

cd /home1/doyoonkim/projects/elsa
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

DENSE_MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
python -u debug_gen_format.py "$DENSE_MODEL" 8192 32
