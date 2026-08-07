#!/bin/bash
#SBATCH --job-name=alps_sparse_ntp_1.7b_pgd
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/alps_sparse_ntp_1.7b_pgd_%j.out
exec 2>&1

# Same recipe as slurm_alps_sparse_ntp_qwen3_1.7b.sh (ALPS s50pct -> fixed-mask
# NTP-only sparse SFT, Qwen3-1.7B, OT80/FW20), but with the base optimizer
# swapped from AdamW to ActivationMetricProjectedSGD (--gmp_base_optimizer=
# activation_metric_pgd, see lib/activation_metric_projected_sgd.py) -- an
# online per-step activation-covariance-projected gradient step, ported from
# opt_baseline_run where it reportedly worked well. LR is the free parameter
# under search here since this optimizer's step scale isn't comparable to
# AdamW's; everything else matches job 40380 exactly.
#
# Usage: sbatch slurm_alps_sparse_ntp_qwen3_1.7b_pgd.sh <LR> [SPARSITY] [LR_SCHEDULER]
# e.g.: sbatch slurm_alps_sparse_ntp_qwen3_1.7b_pgd.sh 0.01
#       sbatch slurm_alps_sparse_ntp_qwen3_1.7b_pgd.sh 0.003 0.5 cosine

LR=${1:?"Usage: sbatch slurm_alps_sparse_ntp_qwen3_1.7b_pgd.sh <LR> [SPARSITY] [LR_SCHEDULER] [GRAD_CKPT]"}
SPARSITY=${2:-0.5}
LR_SCHEDULER=${3:-cosine}
GRAD_CKPT=${4:-false}

ALPS_MODEL="cosmos1030/alps-s50pct_20260802_055049"
REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
DATA_PATH="${REPO_ROOT}/data/ot3_fineweb_200k_qwen3_train.jsonl"

# Must go through `conda activate` (not just invoke the env's python binary by
# absolute path) -- flash_attn's compiled CUDA extension resolves against
# whatever glibc LD_LIBRARY_PATH exposes first, and without activation this
# node picks up a system glibc that's missing symbol version GLIBC_2.32,
# crashing with "undefined symbol: __libc_single_threaded" on model load.
source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

LOCAL_JOB_BASE="/tmp/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/models"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_HOME=/home/shared/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_HOST_IP=127.0.0.1

echo "=== ALPS -> Sparse SFT NTP-only Qwen3-1.7B s${SPARSITY} optimizer=activation_metric_pgd lr=${LR} lr_scheduler=${LR_SCHEDULER} (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$ALPS_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd "$REPO_ROOT"

python main.py \
    --model="$ALPS_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --gmp_base_optimizer=activation_metric_pgd \
    --gmp_gradient_checkpointing=${GRAD_CKPT} \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --seqlen=2048 \
    --gmp_ntp_lambda=1.0 \
    --gmp_kd_lambda=0.0 \
    --gmp_save_path="${REPO_ROOT}/models" \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
