#!/bin/bash
#SBATCH --job-name=build_ot3fw_qwen3_reasoningqat
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/build_ot3fw_qwen3_reasoningqat_%j.out
exec 2>&1

# Replicates the ReasoningQAT (github.com/yasu0001/ReasoningQAT) data recipe
# exactly, verified line-by-line against datautils_block.py:
#   - OpenThoughts3: keep only rows that render to >= seqlen(2048) tokens
#     (discard anything shorter), front-truncate the rest at train time.
#   - FineWeb-Edu: concatenate raw docs (bos+text+eos) into a running buffer
#     until it reaches seqlen tokens, then cut the first seqlen tokens as one
#     packed sample (--pack_fineweb) -- NOT a per-document length filter.
# Every kept OT3 row is guaranteed to get truncated before reaching its
# final answer (front-truncation strategy, seqlen fixed by OPD/vLLM memory
# constraints at 2048 unlike ReasoningQAT's own 8192 distillation seqlen) --
# this is a deliberate side-by-side comparison arm against THINKSTRIP-200K,
# not expected to fix the truncation/final-answer problem itself.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "=== Building OT80/FW20 dataset for Qwen3 (ReasoningQAT recipe: min_tokens=2048, no think-strip) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"

if ! curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/build_ot3_fineweb_dataset.py \
    --nsamples 200000 \
    --out_path data/ot3_fineweb_200k_qwen3_reasoningqat.jsonl \
    --model_path "$MODEL" \
    --seed 42 \
    --min_tokens 2048 \
    --seqlen 2048 \
    --pack_fineweb \
    --num_proc ${SLURM_CPUS_PER_TASK:-24}

echo "=== EXIT: $? ==="
