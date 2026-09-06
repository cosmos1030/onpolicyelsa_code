#!/bin/bash
#SBATCH --job-name=tr_opkd_1.7b_24
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPKD (Dense teacher) Qwen3-1.7B, 2:4 semi-structured sparsity
# Same recipe as wandb jmagj6yb (gmp_s50pct_onpol_lam1_tok256_lr0.0001_4096steps,
# reasoning_qwen3_1.7b project) but sparsity_type=2:4 instead of unstructured,
# and gmp_steps=8192 (unified with the 4B run).
# Usage: sbatch slurm_gmp_tr_opkd_dense_qwen3_1.7b_24.sh <KL_THRESHOLD>

KL_THRESHOLD=${1:?"Usage: sbatch slurm_gmp_tr_opkd_dense_qwen3_1.7b_24.sh <KL_THRESHOLD>"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/slurm"

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

echo "=== TR-GMP NTP+KD+OPKD(Dense) Qwen3-1.7B 2:4 kl=${KL_THRESHOLD} (same recipe as jmagj6yb) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=0.5 \
    --sparsity_type=2:4 \
    --do_gmp=true \
    --gmp_steps=8192 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=32 \
    --gmp_fisher_beta=0.999 \
    --gmp_max_seq_len=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_kd_lambda=1 \
    --gmp_onpolicy_kd_lambda=1 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.35 \
    --gmp_prompt_path="$DATA_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.005 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42
MAIN_EXIT=$?
echo "=== main.py exit: $MAIN_EXIT ==="
[ $MAIN_EXIT -ne 0 ] && { echo "FATAL: main.py failed, skipping rundb register"; exit $MAIN_EXIT; }

# === rundb: register results ===
_WBID=$(cat "$WANDB_RUN_ID_OUTPUT" 2>/dev/null | tr -d '\n')
if [ -n "$_WBID" ]; then
    cd /home1/doyoonkim/projects/elsa/scripts
    $PYTHON rundb/cli.py register \
        --model qwen3_1.7b \
        --sparsities "0.5" \
        --badge tropkd_24 \
        --name "TR-GMP NTP+KD + OPKD (Dense, 2:4 semi-structured)" \
        --sub "kl=${KL_THRESHOLD}_lam1_tok256_lr1e-4_8192steps" \
        --wbid "$_WBID" 2>&1 || echo "rundb register failed (non-fatal)"
else
    echo "WARNING: wandb run ID not found, skipping rundb register"
fi

echo "##### END #####"
exit $MAIN_EXIT
