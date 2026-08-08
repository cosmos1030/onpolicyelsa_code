#!/bin/bash
#SBATCH --job-name=eval_verify_dwup_s70
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_verify_dwup_s70_%j.out
exec 2>&1

# Standalone re-verification eval for cosmos1030/qwen3-4b-gmp-tr-dwup-s70pct.
# No wandb training run could be located for this checkpoint (searched
# reasoning_qwen3_4b for "dwup"+s70 and for hub_model_id match -- nothing
# found; closest s70/onpol candidates from the same window all crashed at
# step ~11). Confirmed sparsity IS exactly 70.0% via direct safetensors
# inspection. Previously reported numbers (math500=67, lcb=9.7, gpqa=26.8,
# ifeval=37.5) have no verifiable source -- re-running lighteval fresh here
# as a brand-new wandb run (no --wandb_run_id to resume) to confirm.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL_REPO="cosmos1030/qwen3-4b-gmp-tr-dwup-s70pct"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=0

echo "=== Re-verifying eval for $MODEL_REPO (s70, provenance unknown) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_REPO" \
    --wandb_project reasoning_qwen3_4b \
    --run_name verify_gmp_tr_dwup_s70pct \
    --method gmp_tr \
    --sparsity 0.7 \
    --gpu_util 0.85 \
    --hub_model_id "$MODEL_REPO" \
    --out_base "$LOCAL_JOB_BASE/eval_verify"

echo "##### END #####"
