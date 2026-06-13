#!/bin/bash
#SBATCH --job-name=dbg_fmt_s50
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dbg_fmt_s50_%j.out

exec 2>&1

cd /home1/doyoonkim/projects/elsa
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

S50_MODEL="/home1/doyoonkim/projects/elsa/models/gmp_s50pct_lr0.0001_cgrpo_0521_1252"
python -u debug_gen_format.py "$S50_MODEL" 8192 32
