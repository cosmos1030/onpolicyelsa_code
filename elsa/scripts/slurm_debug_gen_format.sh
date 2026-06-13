#!/bin/bash
#SBATCH --job-name=debug_gen_fmt
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_debug_gen_fmt_%j.out
#SBATCH --error=slurm_debug_gen_fmt_%j.err

cd /home1/doyoonkim/projects/elsa

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

DENSE_MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
S50_MODEL="/home1/doyoonkim/projects/elsa/models/gmp_s50pct_lr0.0001_cgrpo_0521_1252"
S70_MODEL="/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_cgrpo_0521_1531"

MAX_NEW_TOKENS=4096
N_SAMPLES=64

echo "============================================================"
echo "  [1/3] Dense model (DeepSeek-R1-Distill-Qwen-1.5B)"
echo "============================================================"
python debug_gen_format.py "$DENSE_MODEL" $MAX_NEW_TOKENS $N_SAMPLES

echo ""
echo "============================================================"
echo "  [2/3] Sparse s50% model (cgrpo_0521_1252)"
echo "============================================================"
python debug_gen_format.py "$S50_MODEL" $MAX_NEW_TOKENS $N_SAMPLES

echo ""
echo "============================================================"
echo "  [3/3] Sparse s70% model (cgrpo_0521_1531)"
echo "============================================================"
python debug_gen_format.py "$S70_MODEL" $MAX_NEW_TOKENS $N_SAMPLES
