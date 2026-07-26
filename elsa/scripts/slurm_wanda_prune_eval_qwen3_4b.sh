#!/bin/bash
#SBATCH --job-name=wanda_qwen3_4b
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/wanda_qwen3_4b_%j.out
exec 2>&1

# Wanda prune + full eval for Qwen3-4B → reasoning_qwen3_4b
# Usage: sbatch slurm_wanda_prune_eval_qwen3_4b.sh <SPARSITY> [NSAMPLES=128]

SPARSITY=${1:?"Usage: sbatch slurm_wanda_prune_eval_qwen3_4b.sh <SPARSITY> [NSAMPLES]"}
NSAMPLES=${2:-128}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
SAVE_PATH="/home1/doyoonkim/projects/elsa/models/qwen3_4b_wanda_s${SPARSITY_PCT}pct_n${NSAMPLES}"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

echo "=== Wanda Qwen3-4B (s${SPARSITY_PCT}%) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "SPARSITY=$SPARSITY  NSAMPLES=$NSAMPLES"
echo "SAVE_PATH=$SAVE_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/RAC/open-r1-main

$PYTHON src/open_r1/prune_and_eval.py \
    --model_path "$MODEL" \
    --method wanda \
    --sparsity "$SPARSITY" \
    --nsamples "$NSAMPLES" \
    --seqlen 2048 \
    --save_path "$SAVE_PATH" \
    --wandb_project reasoning_qwen3_4b \
    --wandb_name "wanda_s${SPARSITY_PCT}"

# === rundb: register result ===
/home1/doyoonkim/miniconda3/envs/rac/bin/python /home1/doyoonkim/projects/elsa/scripts/rundb/cli.py add-run \
    --model qwen3_4b \
    --sparsity "$SPARSITY" \
    --badge wanda \
    --name "Wanda ${SPARSITY_PCT}%" \
    --sub "one-shot · no fine-tuning" \
    --wandb-name "wanda_s${SPARSITY_PCT}" \
    --project reasoning_qwen3_4b \
    --sync 2>&1 || echo "rundb add-run failed (non-fatal)"
# ==========================================

# === git push results_db.json ===
_GIT_ROOT="/home1/doyoonkim/projects"
git -C "$_GIT_ROOT" add elsa/scripts/results_db.json
if ! git -C "$_GIT_ROOT" diff --cached --quiet; then
    git -C "$_GIT_ROOT" commit -m "chore: auto-update results_db (job ${SLURM_JOB_ID})" \
        && git -C "$_GIT_ROOT" push 2>&1 \
        || echo "WARNING: git push failed (non-fatal)"
fi
# ================================

echo "##### END #####"
