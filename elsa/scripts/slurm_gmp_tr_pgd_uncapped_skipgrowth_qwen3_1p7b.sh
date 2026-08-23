#!/bin/bash
#SBATCH --job-name=tr_pgd_skipgrowth_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_pgd_skipgrowth_1.7b_%j.out
exec 2>&1

# TR-GMP does the mask GROWTH (KL-gated, every gmp_mask_interval steps, same
# as the plain TR-GMP baseline), but PGD's per-step reprojection during the
# "recovery" steps between growth events is completely uncapped -- no
# kl_share/kl_budget/max_swap_frac, so trust-region KL only governs when/how
# much the mask GROWS, not how PGD reprojects it in between.
#
# --gmp_pgd_skip_growth_step=true is the key difference from the earlier
# rerun_ot80fw20/slurm_gmp_tr_ntpkd_opkd_qwen3_1.7b_pgd.sh: PGD has no gate
# on mask_interval (it fires every single step), so on the EXACT step growth
# just fired, PGD used to immediately re-touch the mask again in the same
# step -- the model never actually trained even one step under the mask
# TR-GMP just decided on before PGD overwrote parts of it. This flag skips
# PGD specifically on step % mask_interval == 0, so the grown mask survives
# through that step's optimizer update untouched, and PGD only starts
# reprojecting again from the next step onward (mask_interval-1 "recovery"
# steps per window).
#
# Usage: sbatch slurm_gmp_tr_pgd_uncapped_skipgrowth_qwen3_1p7b.sh <SPARSITY> <KL_THRESHOLD> [MASK_INTERVAL] [LR] [STEPS] [SPARSITY_TYPE]

SPARSITY=${1:?"Usage: <SPARSITY> <KL_THRESHOLD> [MASK_INTERVAL] [LR] [STEPS] [SPARSITY_TYPE]"}
KL_THRESHOLD=${2:-0.02}
MASK_INTERVAL=${3:-32}
LR=${4:-1e-4}
STEPS=${5:-2048}
SPARSITY_TYPE=${6:-unstructured}
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

echo "=== TR-GMP growth (kl=${KL_THRESHOLD}, mi=${MASK_INTERVAL}) + PGD uncapped during recovery, skip-growth-step=true, Qwen3-1.7B s${SPARSITY_PCT} lr=${LR} steps=${STEPS} ==="
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
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b_nostrip8192 \
    --run_name_suffix="tr_pgd_uncapped_skipgrowth_s${SPARSITY_PCT}_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}_${SPARSITY_TYPE//:/to}_globalscope_$(basename "$DATA_PATH" .jsonl)" \
    --seed=42

echo "##### END #####"
