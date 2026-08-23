#!/bin/bash
#SBATCH --job-name=pgd_notr_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/pgd_notr_1.7b_%j.out
exec 2>&1

# Full eval/save/push version of slurm_debug_pgd_convergence_1p7b.sh: TR-GMP
# disabled entirely (--gmp_tr_enabled=false), mask_interval=1 +
# sparse_train_steps=steps-1 forces the cubic ramp into a single step-1 snap
# straight to the final target sparsity, then PGD (--gmp_pgd=true, uncapped
# -- no kl_share/kl_budget/max_swap_frac) is the ONLY thing moving the mask
# for the remaining ~steps-1 steps, at a fixed keep-count throughout (uses
# the symmetric k=min(revive_cand,prune_cand) cap fix in gmp_trainer.py, so
# sparsity stays exactly fixed instead of drifting).
#
# Usage: sbatch slurm_pgd_notr_uncapped_qwen3_1p7b.sh <SPARSITY> [STEPS] [LR] [SPARSITY_TYPE]

SPARSITY=${1:?"Usage: <SPARSITY> [STEPS] [LR] [SPARSITY_TYPE]"}
STEPS=${2:-2048}
LR=${3:-1e-4}
SPARSITY_TYPE=${4:-unstructured}
SPARSE_TRAIN_STEPS=$((STEPS - 1))
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
# Disables vLLM's background usage-reporting thread (report_usage() ->
# _report_continuous_usage, a `while True: time.sleep(600)` loop) --
# observed to cause a rare but reproducible interpreter-level crash
# ("Fatal Python error: none_dealloc: deallocating None") deep into
# training (e.g. step ~1650/2048), always right at a vLLM wake_up()
# call. Not needed for a research training loop.
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== TR-GMP disabled, PGD-only (uncapped, no TR/schedule) Qwen3-1.7B s${SPARSITY_PCT} steps=${STEPS} ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=${SPARSITY_TYPE} \
    --do_gmp=true \
    --steps=${STEPS} \
    --gmp_sparse_train_steps=${SPARSE_TRAIN_STEPS} \
    --gmp_dense_warmup_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_mask_interval=1 \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --gmp_pruning_scope=global \
    --seqlen=8192 \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=32 \
    --gmp_onpolicy_max_new_tokens=512 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=false \
    --gmp_growth_schedule=cubic \
    --gmp_pgd=true \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b_nostrip8192 \
    --run_name_suffix="pgd_notr_uncapped_s${SPARSITY_PCT}_lr${LR}_mi1_${SPARSITY_TYPE//:/to}_globalscope_$(basename "$DATA_PATH" .jsonl)" \
    --seed=42

echo "##### END #####"
