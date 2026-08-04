#!/bin/bash
#SBATCH --job-name=build_ot3fw_qwen3
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/build_ot3fw_qwen3_%j.out
exec 2>&1

# Rebuild the OT80/FW20 (OpenThoughts3-1.2M 80% + FineWeb-Edu 20%) unified
# dataset using the Qwen3-1.7B tokenizer's OWN chat template, instead of the
# existing ot3_fineweb_20k.jsonl which was rendered with DeepSeek-R1-Distill-
# Qwen-1.5B's chat template (completely different special tokens — confirmed
# not compatible with Qwen3). This replaces math_220k_cot.jsonl as the
# training data for all elsa ADMM (cubic/trust-region) Qwen3-1.7B experiments,
# per the OT80/FW20 unification decision.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "=== Building OT80/FW20 dataset for Qwen3-1.7B ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"

if ! curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/build_ot3_fineweb_dataset.py \
    --nsamples 200000 \
    --out_path data/ot3_fineweb_200k_qwen3.jsonl \
    --model_path "$MODEL" \
    --seed 42 \
    --num_proc ${SLURM_CPUS_PER_TASK:-32}

echo "=== EXIT: $? ==="
