#!/bin/bash
#SBATCH --job-name=tr_opkd_4b_24_lasso
#SBATCH --partition=H200
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n87,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPKD (Dense teacher) Qwen3-4B, 2:4 semi-structured sparsity
# + plain L1/lasso regularization, constant LR (no warmup/decay), kl=0.01 fixed.
# Same base recipe as slurm_gmp_tr_opkd_dense_qwen3_4b_24_h200.sh, changes:
#   - gmp_lr_schedule=constant with a 128-step linear warmup, then flat lr=1e-4
#   - gmp_l1_lambda=1e-4, gmp_l1_mode=plain, gmp_l1_structured=false (plain L1, not bottom-2-per-group)
# Usage: sbatch slurm_gmp_tr_opkd_dense_qwen3_4b_24_lasso_h200.sh <KL_THRESHOLD>

KL_THRESHOLD=${1:?"Usage: sbatch slurm_gmp_tr_opkd_dense_qwen3_4b_24_lasso_h200.sh <KL_THRESHOLD>"}
L1_LAMBDA=1e-4

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
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

echo "=== TR-GMP NTP+KD+OPKD(Dense) Qwen3-4B 2:4 + lasso(plain,l1=${L1_LAMBDA}) + constant lr, kl=${KL_THRESHOLD} ==="
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
    --gmp_lr_schedule=constant \
    --gmp_lr_warmup_steps=128 \
    --gmp_mask_interval=32 \
    --gmp_fisher_beta=0.999 \
    --gmp_max_seq_len=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$DATA_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.0001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_l1_lambda=${L1_LAMBDA} \
    --gmp_l1_mode=plain \
    --gmp_l1_structured=false \
    --gmp_l1_open_groups_only=true \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b \
    --seed=42
MAIN_EXIT=$?
echo "=== main.py exit: $MAIN_EXIT ==="
[ $MAIN_EXIT -ne 0 ] && { echo "FATAL: main.py failed, skipping rundb register"; exit $MAIN_EXIT; }

# === rundb: register results ===
_WBID=$(cat "$WANDB_RUN_ID_OUTPUT" 2>/dev/null | tr -d '\n')
if [ -n "$_WBID" ]; then
    cd /home1/doyoonkim/projects/elsa/scripts
    $PYTHON rundb/cli.py register \
        --model qwen3_4b \
        --sparsities "0.5" \
        --tier N24 \
        --badge tropkd_24 \
        --name "TR-GMP NTP+KD + OPKD (Dense, 2:4 + plain lasso, open-groups)" \
        --sub "kl=${KL_THRESHOLD}_l1=${L1_LAMBDA}_openonly_constlr1e-4_lam0.33_tok256_8192steps" \
        --wbid "$_WBID" 2>&1 || echo "rundb register failed (non-fatal)"
else
    echo "WARNING: wandb run ID not found, skipping rundb register"
fi

echo "##### END #####"
exit $MAIN_EXIT
