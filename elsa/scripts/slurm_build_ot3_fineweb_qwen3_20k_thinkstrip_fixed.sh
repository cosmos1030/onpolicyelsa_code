#!/bin/bash
#SBATCH --job-name=build_ot3fw_qwen3_20k_thinkstrip_fixed
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/build_ot3fw_qwen3_20k_thinkstrip_fixed_%j.out
exec 2>&1

# 20k-scale rebuild of THINKSTRIP with the double-think-artifact fix applied
# (build_ot3_fineweb_dataset.py's strip logic now strips leading empty
# <think></think> stubs before rfind-ing the real closing tag, instead of
# matching the first </think> it finds -- which was the empty stub's close
# for 61% of rows in the old 200k build, silently defeating the strip for
# those rows entirely). Built at LEGACY-20K's scale (20k rows, ot_ratio=0.8
# default = 16k OT3 + 4k FineWeb, matching OT80/FW20) to directly compare
# strip-success-rate against LEGACY-20K and the old (buggy) THINKSTRIP-200K.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "=== Building OT80/FW20 20k dataset for Qwen3 (think-strip-if-long FIXED, seqlen=2048) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"

if ! curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/build_ot3_fineweb_dataset.py \
    --nsamples 20000 \
    --out_path data/ot3_fineweb_20k_qwen3_thinkstrip_fixed.jsonl \
    --model_path "$MODEL" \
    --seed 42 \
    --seqlen 2048 \
    --strip_think_if_long \
    --num_proc ${SLURM_CPUS_PER_TASK:-24}

echo "=== EXIT: $? ==="
