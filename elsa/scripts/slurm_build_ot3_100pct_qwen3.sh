#!/bin/bash
#SBATCH --job-name=build_ot3_100pct_qwen3
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/build_ot3_100pct_qwen3_%j.out
exec 2>&1

# Replicates ReasoningQAT's Stage 2 (end-to-end distillation) data recipe:
# 100% OpenThoughts3-1.2M, no FineWeb-Edu mixed in at all (--ot_ratio=1.0).
# No length filtering/packing/think-stripping -- plain rendering, same as
# PLAIN-200K's OT3 portion, just at full 100% ratio. Intended for training
# at seqlen=8192 (ReasoningQAT Stage 2's own seqlen), not the 2048 used
# everywhere else in this project.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "=== Building 100% OpenThoughts3 dataset for Qwen3 (ReasoningQAT Stage2 recipe) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"

if ! curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/build_ot3_fineweb_dataset.py \
    --nsamples 200000 \
    --out_path data/ot3_100pct_qwen3.jsonl \
    --model_path "$MODEL" \
    --seed 42 \
    --ot_ratio 1.0 \
    --num_proc ${SLURM_CPUS_PER_TASK:-24}

echo "=== EXIT: $? ==="
