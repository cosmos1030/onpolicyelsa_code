#!/bin/bash
#SBATCH --job-name=build_ot3fw_qwen3_40k_nostrip_8192
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/build_ot3fw_qwen3_40k_nostrip_8192_%j.out
exec 2>&1

# Same as slurm_build_ot3_fineweb_qwen3_40k_thinkstrip_fixed_8192.sh (OT80/FW20,
# seqlen=8192, pack_fineweb) but WITHOUT --strip_think_if_long -- plain
# rendering, matching the 100%-OT3 recipe's approach (which never stripped
# think blocks either) but with FineWeb-Edu mixed back in this time. Still
# uses the fixed _manual_render() (no apply_chat_template call) since that
# path runs regardless of strip_think_if_long -- so this build still avoids
# the double-think collapse artifact even though it isn't attempting to
# shorten any long rows.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

echo "=== Building OT80/FW20 40k dataset for Qwen3 (NO think-strip, seqlen=8192, pack_fineweb) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"

if ! curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/build_ot3_fineweb_dataset.py \
    --nsamples 40000 \
    --ot_ratio 0.8 \
    --out_path data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    --model_path "$MODEL" \
    --seed 42 \
    --seqlen 8192 \
    --pack_fineweb \
    --num_proc ${SLURM_CPUS_PER_TASK:-24}

echo "=== EXIT: $? ==="
