#!/bin/bash
#SBATCH --job-name=tr_opkd_1.7b_24_lasso_fsdp_n42
#SBATCH --partition=A6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:5
#SBATCH --nodelist=n42
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPKD (Dense teacher) Qwen3-1.7B, 2:4 semi-structured sparsity
# + plain L1/lasso + constant LR, FSDP across 4 GPUs + vLLM on a DEDICATED 5th GPU.
# Testing n42 specifically (A6000, 48GB) — previously flagged for extremely slow
# first NCCL collective on multi-GPU jobs (see infra_broken_nodes memory); trying
# again in case it's been fixed since.
#
# global batch size kept identical to the single-GPU recipe (batch=1, grad_accum=8
# -> 8 samples/optimizer-step): with 4 FSDP (data-parallel) ranks, grad_accum is
# reduced to 2 so that world_size(4) * batch_size(1) * grad_accum(2) = 8, same as
# before.
#
# Usage: sbatch slurm_gmp_tr_opkd_dense_qwen3_1.7b_24_lasso_fsdp_a6000_n42.sh <KL_THRESHOLD>

KL_THRESHOLD=${1:?"Usage: sbatch slurm_gmp_tr_opkd_dense_qwen3_1.7b_24_lasso_fsdp_a6000_n42.sh <KL_THRESHOLD>"}
L1_LAMBDA=1e-4

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
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
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== TR-GMP NTP+KD+OPKD(Dense) Qwen3-1.7B 2:4 + lasso(plain,l1=${L1_LAMBDA}) + constant lr, kl=${KL_THRESHOLD}, FSDPx4(A6000/n42) + dedicated vLLM GPU ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MASTER_PORT=$MASTER_PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=${MASTER_PORT} \
    main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=0.5 \
    --sparsity_type=2:4 \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --gmp_steps=8192 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=2 \
    --gmp_lr=1e-4 \
    --gmp_lr_schedule=constant \
    --gmp_lr_warmup_steps=128 \
    --gmp_mask_interval=32 \
    --gmp_fisher_beta=0.999 \
    --gmp_max_seq_len=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_kd_lambda=1 \
    --gmp_onpolicy_kd_lambda=1 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.85 \
    --gmp_prompt_path="$DATA_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.005 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_l1_lambda=${L1_LAMBDA} \
    --gmp_l1_mode=plain \
    --gmp_l1_structured=false \
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
        --tier N24 \
        --badge tropkd_24 \
        --name "TR-GMP NTP+KD + OPKD (Dense, 2:4 + plain lasso, FSDPx4)" \
        --sub "kl=${KL_THRESHOLD}_l1=${L1_LAMBDA}_constlr1e-4_lam1_tok256_8192steps" \
        --wbid "$_WBID" 2>&1 || echo "rundb register failed (non-fatal)"
else
    echo "WARNING: wandb run ID not found, skipping rundb register"
fi

echo "##### END #####"
exit $MAIN_EXIT
