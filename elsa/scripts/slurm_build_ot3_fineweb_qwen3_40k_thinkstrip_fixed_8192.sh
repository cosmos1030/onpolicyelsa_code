#!/bin/bash
#SBATCH --job-name=build_ot3fw_qwen3_40k_thinkstrip_fixed_8192
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/build_ot3fw_qwen3_40k_thinkstrip_fixed_8192_%j.out
exec 2>&1

# 40k-scale OT80/FW20 build for the seqlen=8192 recipe, with both the
# double-think render fix (manual ChatML assembly instead of
# apply_chat_template -- see build_ot3_fineweb_dataset.py's _manual_render)
# and FineWeb-Edu mixed back in (via --pack_fineweb, since raw FineWeb docs
# are much shorter than 8192 and would otherwise waste most of the budget
# individually). --seqlen 8192 here matches the actual training seqlen so
# the strip-vs-keep decision is made against the real truncation budget,
# not the old 2048 default. Sized at 40k (32k OT3 + 8k FineWeb) for headroom
# over the ~16,384 samples (steps=2048 x global_batch=8) a full sweep run
# could actually consume, without the multi-hour build/cache cost of the
# earlier 100k/200k builds.
#
# Direct counterpart to the 100%-OT3 seqlen=8192 sweep (jobs 706571,
# 706657-706667) which used the OLD unfixed render + no FineWeb at all
# (ot_ratio=1.0) -- that combination hit math500 73.0/65.6/43.0 (s50/60/70)
# but wikitext2 PPL regressed vs LEGACY-20K/PLAIN-200K (24.6 vs ~18-19),
# most likely from dropping FineWeb entirely. This build isolates whether
# fixing the render bug AND restoring FineWeb recovers PPL while keeping
# (or improving) the math500 gains.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "=== Building OT80/FW20 40k dataset for Qwen3 (think-strip-if-long FIXED, seqlen=8192, pack_fineweb) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"

if ! curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/build_ot3_fineweb_dataset.py \
    --nsamples 40000 \
    --ot_ratio 0.8 \
    --out_path data/ot3_fineweb_40k_qwen3_thinkstrip_fixed_8192.jsonl \
    --model_path "$MODEL" \
    --seed 42 \
    --seqlen 8192 \
    --strip_think_if_long \
    --pack_fineweb \
    --num_proc ${SLURM_CPUS_PER_TASK:-24}

echo "=== EXIT: $? ==="
