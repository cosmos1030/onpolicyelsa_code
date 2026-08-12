#!/bin/bash
#SBATCH --job-name=alps_1.7b_selfgen
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/local-data/user-data/%u/alps_1.7b_selfgen_%j/slurm_%j.out
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91,n61,n64
exec 2>&1

# Same as slurm_alps_prune_1.7b.sh but calibrated on self-gen CoT traces
# (dense Qwen3-1.7B's OWN completions on the same OT3 prompts, generated via
# RAC's grpo.py --trace_only, mixed 80/20 with FineWeb-Edu by
# build_selfgen_ot3_fineweb_dataset.py) instead of the original OT3 teacher's
# answers -- tests whether calibrating on the model's own distribution
# prunes better than calibrating on a different teacher's text.
#
# Usage: sbatch slurm_alps_prune_1.7b_selfgen.sh <SPARSITY>

SPARSITY=${1:?"Usage: sbatch slurm_alps_prune_1.7b_selfgen.sh <SPARSITY>"}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA="/home1/doyoonkim/projects/elsa/data/selfgen_ot3_fineweb_qwen3_8192_v2.jsonl"
SAVE_BASE="/home1/doyoonkim/projects/elsa/models"
SAVED_MODEL="${SAVE_BASE}/qwen3_1.7b_alps_selfgen_v2_s${SPARSITY_PCT}pct"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/alps_1.7b_selfgen_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/eval_out"

DEBUG_COPY_DIR="/home1/doyoonkim/projects/elsa/logs/alps_1.7b_selfgen_${SLURM_JOB_ID}_debug"
mkdir -p "$DEBUG_COPY_DIR"
copy_log_on_exit() {
    cp "$LOCAL_JOB_BASE/slurm_${SLURM_JOB_ID}.out" "$DEBUG_COPY_DIR/" 2>/dev/null || true
    cp "$LOCAL_JOB_BASE/eval_out/eval_summary.json" "$DEBUG_COPY_DIR/" 2>/dev/null || true
}
trap copy_log_on_exit EXIT

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

echo "=== ALPS Pruning Qwen3-1.7B (self-gen calib) s${SPARSITY_PCT}% ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/ALPS

$PYTHON qwen3_alps.py \
    "$MODEL" \
    ${SPARSITY} \
    --data_path "$DATA" \
    --nsamples 128 \
    --rho 300.0 \
    --seed 42 \
    --save "$SAVED_MODEL" \
    --eval_full \
    --wandb_project reasoning_qwen3_1.7b_nostrip8192 \
    --run_name "alps_selfgen_v2_s${SPARSITY_PCT}pct" \
    --gpu_util 0.9 \
    --tp_size 1 \
    --out_base "$LOCAL_JOB_BASE/eval_out" \
    --profile quick \
    --push_to_hub

EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE ==="
echo "##### END #####"
