#!/bin/bash
#SBATCH --job-name=gmp_pgd_cosine_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/gmp_pgd_cosine_1.7b_%j.out
exec 2>&1

# New PGD design, replacing TR-GMP's adaptive KL-gated growth with a fixed
# cosine sparsity schedule + per-step PGD reprojection under an explicit
# trust-region-style churn cap:
#   - sparsity target follows a cosine ramp (--gmp_growth_schedule=cosine,
#     --gmp_tr_enabled=false), updated every gmp_mask_interval steps, reaching
#     --sparsity_ratio at step (steps - gmp_sparse_train_steps) = 1024 here
#     (half the 2048-step budget; remaining 1024 steps are fixed-mask sparse
#     training).
#   - PGD (--gmp_pgd=true) still reprojects the mask by fresh fisher
#     importance every single step (not just every mask_interval), but now
#     capped at --gmp_pgd_max_swap_frac of total masked params per step
#     (existing flag, previously never actually used in any production run --
#     uncapped PGD was observed churning up to ~2.4% of the ENTIRE 4B model's
#     params in a single step, ~98% of all steps showing nonzero churn).
#     Capping spreads each mask_interval window's target movement across its
#     32 steps instead of applying it all at once at the window boundary.
#
# Usage: sbatch slurm_gmp_pgd_cosine_qwen3_1.7b.sh <SPARSITY> <SWAP_FRAC> [MASK_INTERVAL] [LR] [OPD_GEN_LEN] [DATA_PATH] [SEQLEN] [WANDB_PROJECT]
# e.g.: sbatch slurm_gmp_pgd_cosine_qwen3_1.7b.sh 0.5 0.0008 32 1e-4

SPARSITY=${1:?"Usage: <SPARSITY> <SWAP_FRAC> [MASK_INTERVAL] [LR] [OPD_GEN_LEN] [DATA_PATH] [SEQLEN] [WANDB_PROJECT]"}
SWAP_FRAC=${2:?"Usage: <SPARSITY> <SWAP_FRAC> [MASK_INTERVAL] [LR] [OPD_GEN_LEN] [DATA_PATH] [SEQLEN] [WANDB_PROJECT]"}
MASK_INTERVAL=${3:-32}
LR=${4:-1e-4}
OPD_GEN_LEN=${5:-256}
DATA_PATH=${6:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${7:-8192}
WANDB_PROJECT=${8:-reasoning_qwen3_1.7b_nostrip8192}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"
STEPS=2048
SPARSE_TRAIN_STEPS=1024

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
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== GMP PGD+cosine Qwen3-1.7B s${SPARSITY_PCT} swap_frac=${SWAP_FRAC} mi=${MASK_INTERVAL} lr=${LR} steps=${STEPS} sparse_train_steps=${SPARSE_TRAIN_STEPS} (OT80/FW20 nostrip8192) ==="
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
    --sparsity_type=unstructured \
    --do_gmp=true \
    --steps=${STEPS} \
    --gmp_sparse_train_steps=${SPARSE_TRAIN_STEPS} \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --gmp_pruning_scope=global \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=false \
    --gmp_growth_schedule=cosine \
    --gmp_pgd=true \
    --gmp_pgd_max_swap_frac=${SWAP_FRAC} \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --run_name_suffix="pgd_cosine_s${SPARSITY_PCT}_swap${SWAP_FRAC}_mi${MASK_INTERVAL}_lr${LR}_$(basename "$DATA_PATH" .jsonl)" \
    --seed=42

echo "##### END #####"
