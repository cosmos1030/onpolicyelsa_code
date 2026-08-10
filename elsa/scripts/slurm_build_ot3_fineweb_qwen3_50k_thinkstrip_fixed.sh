#!/bin/bash
#SBATCH --job-name=build_ot3fw_qwen3_50k_thinkstrip_fixed
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/build_ot3fw_qwen3_50k_thinkstrip_fixed_%j.out
exec 2>&1

# 50k-scale THINKSTRIP build with both fixes applied:
#   1. strip logic strips leading empty <think></think> stub(s) before
#      rfind-ing the real closing tag (previous fix, no-op on its own)
#   2. final text is assembled by hand (_manual_render) instead of calling
#      tok.apply_chat_template() -- avoids Qwen3's template unconditionally
#      collapsing any "</think>"-containing assistant content down to a
#      fresh empty <think></think> stub + text-after-last-</think>, which
#      was producing the "double-think" artifact in 61-63% of rows in EVERY
#      previous Qwen3-templated build (PLAIN-200K, THINKSTRIP-200K alike --
#      confirmed not specific to strip_think_if_long). Verified on a 20k
#      build (job 707149): double-think artifact rate dropped from 61-63%
#      to 0%, matching LEGACY-20K (which never had this bug because its
#      lost build script used DeepSeek's chat template, which concatenates
#      raw content verbatim with no such collapsing behavior).

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "=== Building OT80/FW20 50k dataset for Qwen3 (think-strip-if-long FIXED, seqlen=2048) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"

if ! curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/build_ot3_fineweb_dataset.py \
    --nsamples 50000 \
    --out_path data/ot3_fineweb_50k_qwen3_thinkstrip_fixed.jsonl \
    --model_path "$MODEL" \
    --seed 42 \
    --seqlen 2048 \
    --strip_think_if_long \
    --num_proc ${SLURM_CPUS_PER_TASK:-24}

echo "=== EXIT: $? ==="
