#!/bin/bash
#SBATCH --job-name=pgd_recdiag_probe
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91,n87,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/pgd_recdiag_probe_%j.out
exec 2>&1

# Probe run for --gmp_pgd_recovery_diag (see main.py flag docstring and
# gmp_trainer.py's "pgd_recovery_diag" state block / _pgd_diag_nll). Purpose:
# get a per-projection-event (rho_t, D_proj, support-turnover) time series
# out of an otherwise-unmodified fixed-delta gmp_pgd_grow_to_target run, to
# check whether projection KL (pgd/kl_at_k_actual, already logged) predicts
# low recovery ratio (pgd/recovery_diag/rho, new). Read-only diagnostic: the
# actual applied mask each step is completely unaffected by this flag, only
# two small extra forward passes per pgd_interval-spaced projection event.
#
# Config matches the production s50/kl_budget=0.02/lr=5e-5 recipe (job
# 823372, OT80/FW20 data) that reached sparsity=0.5 by step ~120 and then
# sat in post-target maintenance -- steps=400 here covers the full growth
# ramp plus ~35 more maintenance-phase projection windows (at
# pgd_interval=8), enough for a first correlation read across both regimes.
# save_model/push_to_hub/eval all off -- this run only needs to exist long
# enough to produce the wandb log stream, not a usable checkpoint.

SPARSITY=${1:-0.5}
KL_BUDGET=${2:-0.02}
MASK_INTERVAL=${3:-32}
LR=${4:-5e-5}
PGD_INTERVAL=${5:-8}
STEPS=${6:-400}
WANDB_PROJECT=${7:-pgd_recovery_diag_probe}

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
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== pgd_recovery_diag probe: Qwen3-1.7B s${SPARSITY} kl_budget=${KL_BUDGET} lr=${LR} pgd_interval=${PGD_INTERVAL} steps=${STEPS} (OT80/FW20) ==="
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
    --kd_nsamples=256 \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --steps=${STEPS} \
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
    --gmp_kl_chunk_size=2048 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=${MASK_INTERVAL} \
    --gmp_onpolicy_max_new_tokens=512 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=false \
    --gmp_pruning_end_ratio=0.0 \
    --gmp_pgd=true \
    --gmp_pgd_grow_to_target=true \
    --gmp_pgd_kl_budget=${KL_BUDGET} \
    --gmp_pgd_kl_calib_size=4 \
    --gmp_pgd_interval=${PGD_INTERVAL} \
    --gmp_pgd_recovery_diag=true \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --run_name_suffix="recdiag_probe_s${SPARSITY}_klb${KL_BUDGET}_lr${LR}_pgdi${PGD_INTERVAL}_mi${MASK_INTERVAL}" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
