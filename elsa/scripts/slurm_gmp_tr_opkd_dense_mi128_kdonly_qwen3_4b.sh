#!/bin/bash
#SBATCH --job-name=tr_kd_opkd_mi128_4b
#SBATCH --partition=H200-PCIe-ZT
#SBATCH --qos=zt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n76,n77,n80,n87,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_kd_opkd_mi128_4b_%j.out
exec 2>&1

# TR-GMP KD+OPKD (Dense teacher, no NTP) Qwen3-4B, mask_interval=128
# kd_lambda=0.5, onpolicy_kd_lambda=0.5, ntp_lambda=0.0
# Usage: sbatch slurm_gmp_tr_opkd_dense_mi128_kdonly_qwen3_4b.sh <KL_THRESHOLD>

KL_THRESHOLD=${1:?"Usage: sbatch slurm_gmp_tr_opkd_dense_mi128_kdonly_qwen3_4b.sh <KL_THRESHOLD>"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"

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

echo "=== TR-GMP KD+OPKD(Dense) Qwen3-4B kl=${KL_THRESHOLD} mi=128 (milestones: s50/s60/s70) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=0.7 \
    --do_gmp=true \
    --gmp_steps=32768 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=128 \
    --gmp_fisher_beta=0.999 \
    --gmp_max_seq_len=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=0.0 \
    --gmp_kd_lambda=0.5 \
    --gmp_onpolicy_kd_lambda=0.5 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$DATA_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.005 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_milestone_sparsities="0.5,0.6,0.7" \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b \
    --seed=42

# === rundb: register milestone results ===
_WBID=$(cat "$WANDB_RUN_ID_OUTPUT" 2>/dev/null | tr -d '\n')
if [ -n "$_WBID" ]; then
    cd /home1/doyoonkim/projects/elsa/scripts
    /home1/doyoonkim/miniconda3/envs/rac/bin/python rundb/cli.py register \
        --model qwen3_4b \
        --sparsities "0.5,0.6,0.7" \
        --badge tropkd \
        --name "TR-GMP KD + OPKD (Dense)" \
        --sub "kl=${KL_THRESHOLD}_mi128" \
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
