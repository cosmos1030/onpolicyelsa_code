#!/bin/bash
#SBATCH --job-name=elsa_dynbarrier_smoketest_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/elsa_dynbarrier_smoketest_1.7b_%j/slurm_%j.out
exec 2>&1

# Smoke test for the Dynamic Barrier ADMM coefficient (lib/optimizers.py
# ADMMOptimizer.dynamic_barrier): replaces the fixed/scheduled admm_lmda
# penalty with a per-step closed-form lambda_k = max((phi_k - q.r)/||r||^2, 0)
# that guarantees the ADMM residual shrinks toward a per-interval target
# c_t = barrier_beta * g_start, while otherwise staying as close as possible
# to the raw KD gradient. No SGD validation step (saliency needs Adam's
# exp_avg_sq state, so SGD isn't usable here) -- straight to Adam with raw
# gradients, per the approved plan.
#
# Mask is FIXED at the final sparsity from step 0 (admm_tr_z_proj=false, no
# growth schedule) -- matches the KD-only reference run w5ng5ddy exactly
# (same admm_interval/lr/kd_lambda/etc), the only difference being
# admm_dynamic_barrier=true replacing w5ng5ddy's admm_lmda_schedule_mode=
# cosine fixed schedule. A first attempt at this script mistakenly reused a
# cosine-z-GROWTH config (mask target moving every interval), which made the
# ADMM residual balloon for reasons unrelated to the barrier mechanism itself
# (the projection target never stopped moving) -- fixed here so the smoke
# test isolates the barrier's actual behavior under a stationary target.
# Intentionally short (steps=256, no eval/save/push) -- the only goal is to
# watch barrier/lambda and barrier/residual in wandb and confirm the residual
# actually tracks toward barrier/target_c without lambda pinning at 0 or
# blowing up against admm_barrier_lambda_max, before committing to a full run.
#
# Usage: sbatch slurm_elsa_dynamic_barrier_smoketest_1.7b.sh <SPARSITY> [ALPHA] [BETA] [LAMBDA_MAX]
# e.g.: sbatch slurm_elsa_dynamic_barrier_smoketest_1.7b.sh 0.5 0.5 0.8 100.0

SPARSITY=${1:?"Usage: sbatch slurm_elsa_dynamic_barrier_smoketest_1.7b.sh <SPARSITY> [ALPHA] [BETA] [LAMBDA_MAX]"}
ALPHA=${2:-0.5}
BETA=${3:-0.8}
LAMBDA_MAX=${4:-100.0}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_train.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/elsa_dynbarrier_smoketest_1.7b_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

DEBUG_LOG_COPY="/home1/doyoonkim/projects/elsa/logs/elsa_dynbarrier_smoketest_1.7b_${SLURM_JOB_ID}_last.out"
mkdir -p /home1/doyoonkim/projects/elsa/logs
copy_log_on_exit() { cp "$LOCAL_JOB_BASE/slurm_${SLURM_JOB_ID}.out" "$DEBUG_LOG_COPY" 2>/dev/null || true; }
trap copy_log_on_exit EXIT

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export WANDB_RUN_ID_OUTPUT="/home1/doyoonkim/projects/elsa/logs/handoff_${SLURM_JOB_ID}_wandb_run_id.txt"
export MODEL_PATH_OUTPUT="/home1/doyoonkim/projects/elsa/logs/handoff_${SLURM_JOB_ID}_model_path.txt"
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/tmp
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== Dynamic Barrier smoke test, ELSA KD-only (mask FIXED at s${SPARSITY_PCT} from step 0) Qwen3-1.7B, alpha=${ALPHA} beta=${BETA} lambda_max=${LAMBDA_MAX}, 1xA100-80GB (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=1 --master_port=${MASTER_PORT} main.py \
    --model="$MODEL" \
    --data_path="$DATA_PATH" \
    --dataset=mixed_cot \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --steps=256 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=8 \
    --admm_use_fsdp=false \
    --admm_dynamic_barrier=true \
    --admm_barrier_alpha=${ALPHA} \
    --admm_barrier_beta=${BETA} \
    --admm_barrier_lambda_max=${LAMBDA_MAX} \
    --admm_tr_z_proj=false \
    --lr=1e-4 \
    --lr_scheduler=linear \
    --lr_warmup_steps=0 \
    --seqlen=2048 \
    --admm_base_optimizer=adamw \
    --admm_beta1=0.9 \
    --admm_beta2=0.999 \
    --admm_projection_mode=momentum \
    --admm_interval=32 \
    --admm_precision=bf16 \
    --admm_dual_dtype=fp32 \
    --admm_split_dtype=fp32 \
    --do_offpolicy_kd_admm=true \
    --kd_lambda=1.0 \
    --kd_ntp_lambda=0.0 \
    --kd_topk=0 \
    --kd_use_vllm=false \
    --save_model=false \
    --eval_math500=false \
    --eval_zero_shot=false \
    --eval_full_bench=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42 \
    --push_to_hub=false

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
