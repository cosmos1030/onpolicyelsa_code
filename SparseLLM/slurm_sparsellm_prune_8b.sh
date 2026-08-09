#!/bin/bash
#SBATCH --job-name=sparsellm_8b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/local-data/user-data/%u/sparsellm_8b_%j/slurm_%j.out
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91,n61,n64
exec 2>&1

# SparseLLM pruning + full eval for Qwen3-8B
# Usage: sbatch slurm_sparsellm_prune_8b.sh <SPARSITY>

SPARSITY=${1:?"Usage: sbatch slurm_sparsellm_prune_8b.sh <SPARSITY>"}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DATA="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"
SAVE_BASE="/home1/doyoonkim/projects/elsa/models"
SAVED_MODEL="${SAVE_BASE}/qwen3_8b_sparsellm_s${SPARSITY_PCT}pct"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/sparsellm_8b_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/eval_out"

# Node access is revoked once the job ends, so /local-data logs/results become
# permanently unreachable if something (network outage, wandb failure) goes
# wrong — copy the log and eval_summary.json to NFS once on exit (small, one-
# time write, not continuous I/O) so post-mortem/recovery is possible.
DEBUG_COPY_DIR="/home1/doyoonkim/projects/elsa/logs/sparsellm_8b_${SLURM_JOB_ID}_debug"
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

echo "=== SparseLLM Pruning Qwen3-8B s${SPARSITY_PCT}% ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/SparseLLM

$PYTHON qwen3_main.py \
    --model "$MODEL" \
    --data_path "$DATA" \
    --nsamples 128 \
    --seqlen 2048 \
    --sparsity ${SPARSITY} \
    --seed 42 \
    --save "$SAVED_MODEL" \
    --eval_full \
    --wandb_project reasoning_qwen3_8b \
    --run_name "sparsellm_s${SPARSITY_PCT}pct" \
    --gpu_util 0.9 \
    --tp_size 1 \
    --out_base "$LOCAL_JOB_BASE/eval_out" \
    --push_to_hub

EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE ==="
echo "##### END #####"
