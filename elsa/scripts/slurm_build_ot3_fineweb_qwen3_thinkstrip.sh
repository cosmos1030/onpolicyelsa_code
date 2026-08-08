#!/bin/bash
#SBATCH --job-name=build_ot3fw_qwen3_thinkstrip
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/build_ot3fw_qwen3_thinkstrip_%j.out
exec 2>&1

# Rebuild OT80/FW20 with the think-strip fallback: rows that exceed --seqlen
# tokens as rendered have their <think>...</think> block dropped, keeping only
# the model's own post-think write-up, instead of being silently front-
# truncated by main.py at train time (which was cutting off the final answer
# in ~26% of truncated rows on the plain ot3_fineweb_200k_qwen3_train.jsonl).
# Reverse-engineered from and verified byte-for-byte against the old
# (pre-08-04, undocumented/lost) ot3_fineweb_20k.jsonl build -- this
# reproduces its truncation profile almost exactly in a 3000-sample dry run
# (51.9% still over seqlen=2048 vs the old dataset's measured 51.0%).
# No length-based filtering happens anywhere, so domain mix (math/code/
# science) is unaffected -- only content length changes for the rows that
# needed it.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "=== Building OT80/FW20 dataset for Qwen3 (think-strip-if-long, seqlen=2048) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"

if ! curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/build_ot3_fineweb_dataset.py \
    --nsamples 200000 \
    --out_path data/ot3_fineweb_200k_qwen3_thinkstrip.jsonl \
    --model_path "$MODEL" \
    --seed 42 \
    --seqlen 2048 \
    --strip_think_if_long \
    --num_proc ${SLURM_CPUS_PER_TASK:-24}

echo "=== EXIT: $? ==="
