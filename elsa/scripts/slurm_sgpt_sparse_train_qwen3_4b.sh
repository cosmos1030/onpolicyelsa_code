#!/bin/bash
#SBATCH --job-name=sgpt_sft_4b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/sgpt_sft_4b_%j.out
exec 2>&1

# SparseGPT → sparse SFT (NTP+KD, fixed mask) for Qwen3-4B
# Loads pre-pruned SparseGPT checkpoint, fixes mask from zero pattern,
# runs 2048 steps of NTP+KD sparse fine-tuning, then evaluates.
# Usage: sbatch slurm_sgpt_sparse_train_qwen3_4b.sh <SPARSITY>
#   e.g. sbatch slurm_sgpt_sparse_train_qwen3_4b.sh 0.5

SPARSITY=${1:?"Usage: sbatch slurm_sgpt_sparse_train_qwen3_4b.sh <SPARSITY>"}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
SPARSE_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_4b_sparsellm_s${SPARSITY_PCT}pct"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== SparseGPT Sparse SFT (NTP+KD) Qwen3-4B s${SPARSITY_PCT}% ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "SPARSE_MODEL=$SPARSE_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$SPARSE_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --gmp_steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_max_seq_len=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=1.0 \
    --gmp_kd_lambda=0.0 \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b \
    --seed=42

# === rundb: register result ===
_WBID=$(cat "$WANDB_RUN_ID_OUTPUT" 2>/dev/null | tr -d '\n')
if [ -n "$_WBID" ]; then
    cd /home1/doyoonkim/projects/elsa/scripts
    /home1/doyoonkim/miniconda3/envs/rac/bin/python rundb/cli.py register \
        --model qwen3_4b \
        --sparsities "${SPARSITY}" \
        --badge sgpt_sft \
        --name "SparseGPT + Sparse SFT (NTP+KD)" \
        --sub "2048 steps · fixed mask" \
        --wbid "$_WBID" 2>&1 || echo "rundb register failed (non-fatal)"
else
    echo "WARNING: wandb run ID not found, skipping rundb register"
fi
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
