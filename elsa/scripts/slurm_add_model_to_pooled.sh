#!/bin/bash
#SBATCH --job-name=add_pooled
#SBATCH --partition=A100-80GB,4A100
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# Add one model's rollouts to an existing pooled.npz. See add_model_to_pooled.py.
# Usage: sbatch slurm_add_model_to_pooled.sh <STATES_DIR> <LABEL> <MODEL_PATH>

STATES_DIR=${1:?"<STATES_DIR> <LABEL> <MODEL_PATH>"}
LABEL=${2:?"<LABEL> e.g. noopd:s50"}
MODEL=${3:?"<MODEL_PATH>"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
DENSE="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
[ -z "${LOCAL_JOB_BASE:-}" ] && LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/slurm"
LOG=/home1/doyoonkim/projects/elsa/logs/policy_divergence/add_pooled_${SLURM_JOB_ID}.out
LOCAL_LOG="$LOCAL_JOB_BASE/slurm/add_pooled_${SLURM_JOB_ID}.out"
( while true; do cp "$LOCAL_LOG" "$LOG" 2>/dev/null || true; sleep 30; done ) &
M=$!
trap 'kill $M 2>/dev/null; cp "$LOCAL_LOG" "$LOG" 2>/dev/null || true' EXIT

export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

echo "=== add $LABEL to $STATES_DIR ===  NODE=$(hostname) JOB=$SLURM_JOB_ID"
cd /home1/doyoonkim/projects/elsa
$PYTHON scripts/add_model_to_pooled.py \
    --states_dir "$STATES_DIR" --dense_model "$DENSE" \
    --label "$LABEL" --model_path "$MODEL"
echo "=== EXIT: $? ==="
