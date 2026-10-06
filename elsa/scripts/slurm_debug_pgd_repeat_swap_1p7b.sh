#!/bin/bash
#SBATCH --job-name=dbg_pgd_repeat_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19,n56
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/dbg_pgd_repeat_1.7b_%j.out
exec 2>&1

# Diagnostic-only (no eval/save/push): same TR growth (kl-gated, mask_interval)
# + PGD-uncapped-during-recovery + skip-growth-step setup as
# slurm_gmp_tr_pgd_uncapped_skipgrowth_qwen3_1p7b.sh, but with
# --gmp_pgd_debug_repeat_swap=true: logs pgd/repeat_swap_frac each step --
# what fraction of this step's revive/prune flips are positions that ALSO
# flipped within the last gmp_pgd_debug_repeat_window (default 5) steps.
# Tests whether the within-window churn growth (observed: swap count climbs
# steadily from growth event to growth event, e.g. 190->804 over a 64-step
# window) is the SAME small set of near-threshold weights repeatedly
# swapping back and forth, or a growing set of distinct weights each
# swapping once.
#
# Usage: sbatch slurm_debug_pgd_repeat_swap_1p7b.sh <SPARSITY> <KL_THRESHOLD> [MASK_INTERVAL] [LR] [STEPS]

SPARSITY=${1:?"Usage: <SPARSITY> <KL_THRESHOLD> [MASK_INTERVAL] [LR] [STEPS]"}
KL_THRESHOLD=${2:-0.02}
MASK_INTERVAL=${3:-64}
LR=${4:-1e-4}
STEPS=${5:-512}
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
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== [DEBUG] PGD repeat-swap probe: TR growth (kl=${KL_THRESHOLD}, mi=${MASK_INTERVAL}) + PGD uncapped recovery, skip-growth-step, Qwen3-1.7B s${SPARSITY_PCT} lr=${LR} steps=${STEPS} ==="
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
    --gmp_post_target_steps=0 \
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
    --seqlen=8192 \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=${MASK_INTERVAL} \
    --gmp_onpolicy_max_new_tokens=512 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_pgd=true \
    --gmp_pgd_skip_growth_step=true \
    --gmp_pgd_debug_repeat_swap=true \
    --gmp_pgd_debug_repeat_window=5 \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=debug_pgd_convergence \
    --run_name_suffix="dbg_pgd_repeat_s${SPARSITY_PCT}_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}" \
    --seed=42

echo "##### END #####"
