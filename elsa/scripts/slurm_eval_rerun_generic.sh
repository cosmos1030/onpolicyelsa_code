#!/bin/bash
#SBATCH --job-name=eval_rerun
#SBATCH --partition=H200-PCIe-ZT
#SBATCH --qos=zt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_rerun_%j.out
exec 2>&1

# Generic eval_full.py-only rerun for a checkpoint whose training finished
# but whose own post-training eval crashed (illegal memory access in vLLM --
# seen repeatedly, e.g. jobs 714837, 714896). Logs into the SAME wandb run
# so results land in the original run instead of splitting.
#
# Usage: sbatch slurm_eval_rerun_generic.sh <MODEL_PATH> <SPARSITY> <RUN_NAME> <WANDB_PROJECT> <WANDB_RUN_ID>

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_rerun_generic.sh <MODEL_PATH> <SPARSITY> <RUN_NAME> <WANDB_PROJECT> <WANDB_RUN_ID>"}
SPARSITY=${2:?"..."}
RUN_NAME=${3:?"..."}
WANDB_PROJECT=${4:?"..."}
WANDB_RUN_ID=${5:?"..."}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
LOCAL_JOB_BASE="/local-data/user-data/${USER}/eval_rerun_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/eval_out"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

echo "=== eval_full.py rerun: $MODEL_PATH (sparsity=$SPARSITY) -> wandb run $WANDB_RUN_ID ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "No internet on $(hostname) -- falling back to WANDB_MODE=offline (sync later)."
    export WANDB_MODE=offline
fi

$PYTHON /home1/doyoonkim/projects/elsa/scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_id "$WANDB_RUN_ID" \
    --run_name "$RUN_NAME" \
    --method gmp \
    --sparsity "$SPARSITY" \
    --gpu_util 0.85 \
    --profile quick \
    --out_base "$LOCAL_JOB_BASE/eval_out"

EVAL_EXIT=$?
echo "=== eval_full.py exit: $EVAL_EXIT ==="
echo "##### END #####"
exit $EVAL_EXIT
