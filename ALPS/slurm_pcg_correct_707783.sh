#!/bin/bash
#SBATCH --job-name=pcg_correct_707783
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/pcg_correct_707783_%j.out
exec 2>&1

# Apply an ALPS-style PCG backsolve correction (mask fixed, no ADMM search)
# to job 707783 (TR-GMP NTP+KD+OPKD, s50, lr=5e-5, mi=32, nostrip8192 -- the
# best s50 config from the fisher-saliency sweep, math500=71.0 quick-eval),
# then run quick eval to see if it moves the numbers. Checkpoint was already
# uploaded to HF and deleted locally, so pull it straight from the hub.
# See pcg_correct_gmp_checkpoint.py: each Linear layer's surviving weights
# are re-solved via conjugate gradient to better reconstruct the DENSE
# model's per-layer input/output function, sequentially layer-by-layer
# (same calibration pipeline as qwen3_alps.py).

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

DENSE_MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PRUNED_MODEL="cosmos1030/qwen3-1.7b-tr-gmp-ntpkdopkd-s50pct-lr5e-05-mi32-nostrip8192"
DATA="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
SAVE_DIR="/home1/doyoonkim/projects/elsa/models/707783_pcg_corrected"
LOCAL_JOB_BASE="/local-data/user-data/${USER}/pcg_correct_707783_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/eval_out"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
unset HF_DATASETS_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_HUB_OFFLINE
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "=== PCG correction: TR-GMP NTP+KD+OPKD 1.7B s50 lr5e-5 mi32 (job 707783 checkpoint, from HF) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/ALPS

$PYTHON pcg_correct_gmp_checkpoint.py \
    "$DENSE_MODEL" \
    "$PRUNED_MODEL" \
    --data_path "$DATA" \
    --nsamples 128 \
    --seed 42 \
    --save "$SAVE_DIR"

CORRECT_EXIT=$?
echo "=== PCG correction exit: $CORRECT_EXIT ==="
if [ $CORRECT_EXIT -ne 0 ]; then
    exit $CORRECT_EXIT
fi

echo "=== Running quick eval on corrected checkpoint ==="
$PYTHON /home1/doyoonkim/projects/elsa/scripts/eval_full.py \
    --model_path "$SAVE_DIR" \
    --wandb_project reasoning_qwen3_1.7b_nostrip8192 \
    --run_name 707783_pcg_corrected \
    --method gmp \
    --sparsity 0.5 \
    --gpu_util 0.85 \
    --profile quick \
    --out_base "$LOCAL_JOB_BASE/eval_out"

EVAL_EXIT=$?
echo "=== eval_full.py exit: $EVAL_EXIT ==="
exit $EVAL_EXIT
