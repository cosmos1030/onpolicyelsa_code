#!/bin/bash
#SBATCH --job-name=eval_rerun_714837_lcb
#SBATCH --partition=H200-PCIe-ZT
#SBATCH --qos=zt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=n89,n90,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_rerun_714837_lcb_%j.out
exec 2>&1

# lcb-only rerun for the 4B ALPS-SFT s50 lr=1e-4 checkpoint (gmp_s50pct_lr0.0001_onpol_lmda0.33_20260812_072544,
# wandb run wzavc5nd) -- job 718670 already got math500/gpqa/ifeval/gsm8k
# through cleanly but lcb crashed with the recurring vLLM illegal-memory-access.

MODEL_PATH="/home1/doyoonkim/projects/elsa/models/gmp_s50pct_lr0.0001_onpol_lmda0.33_20260812_072544"
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "No internet on $(hostname) -- falling back to WANDB_MODE=offline (sync later)."
    OFFLINE_WANDB_DIR="/home1/doyoonkim/projects/elsa/logs/wandb_offline/job_${SLURM_JOB_ID}"
    mkdir -p "$OFFLINE_WANDB_DIR"
    export WANDB_DIR="$OFFLINE_WANDB_DIR"
    export WANDB_MODE=offline
fi

echo "=== Re-running lcb only for $MODEL_PATH ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project reasoning_qwen3_4b_nostrip8192 \
    --wandb_run_id wzavc5nd \
    --method gmp \
    --sparsity 0.5 \
    --gpu_util 0.9 \
    --skip_ppl \
    --skip_zeroshot \
    --benchmarks lcb \
    --profile quick

echo "=== Exit code: $? ==="
echo "##### END #####"
