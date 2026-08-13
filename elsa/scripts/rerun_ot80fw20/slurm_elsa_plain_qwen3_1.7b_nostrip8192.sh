#!/bin/bash
#SBATCH --job-name=elsa_plain_1.7b_nostrip8192
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/elsa_plain_1.7b_nostrip8192_%j.out
exec 2>&1

# Plain ELSA (ADMM-projection pruning, NTP-only, no KD) baseline for direct
# comparison against TR-GMP -- matched as closely as possible to the TR-GMP
# recipe used throughout this session: same OT80/FW20 nostrip8192 40k
# calibration+training data, seqlen=8192, steps=2048, global batch=8,
# lr_scheduler=cosine with warmup=256 (TR-GMP's own scheduler/warmup,
# replacing the old replicate_old script's linear/no-warmup). Also fixes
# the previous elsa_plain sweep's admm_z_layerwise being left at its default
# False (global single threshold) -- explicitly true here, matching plain
# ELSA's actual per-layer-uniform default and the "layer-wise" framing this
# baseline is meant to represent. admm_lmda_schedule_mode stays cosine
# (gradually increases lambda over training, per the earlier replicate_old
# precedent) -- only admm_lmda's magnitude is swept, not the schedule mode.
#
# Usage: sbatch slurm_elsa_plain_qwen3_1.7b_nostrip8192.sh <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE]
# e.g.: sbatch slurm_elsa_plain_qwen3_1.7b_nostrip8192.sh 0.5 5e-5 1e-3

SPARSITY=${1:?"Usage: sbatch slurm_elsa_plain_qwen3_1.7b_nostrip8192.sh <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE]"}
LR=${2:?"Usage: sbatch slurm_elsa_plain_qwen3_1.7b_nostrip8192.sh <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE]"}
LMDA=${3:?"Usage: sbatch slurm_elsa_plain_qwen3_1.7b_nostrip8192.sh <SPARSITY> <LR> <LMDA> [SPARSITY_TYPE]"}
SPARSITY_TYPE=${4:-unstructured}

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
SEQLEN=8192
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
# Safe here (unlike the OPKD scripts): this plain-NTP ADMM path never
# touches vLLM, so expandable_segments' CuMemAllocator conflict doesn't apply.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/tmp
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== ELSA plain (ADMM, NTP-only, layer-wise) Qwen3-1.7B s${SPARSITY_PCT} (${SPARSITY_TYPE}) lr=${LR} lmda=${LMDA} (cosine schedule) -- 1xA100-80GB (OT80/FW20 nostrip8192) ==="
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
    --sparsity_type=${SPARSITY_TYPE} \
    --steps=2048 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=8 \
    --admm_use_fsdp=false \
    --admm_lmda=${LMDA} \
    --admm_lmda_schedule_mode=cosine \
    --admm_z_layerwise=true \
    --lr=${LR} \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --seqlen=${SEQLEN} \
    --admm_base_optimizer=adamw \
    --admm_beta1=0.9 \
    --admm_beta2=0.999 \
    --admm_projection_mode=momentum \
    --admm_interval=32 \
    --admm_precision=bf16 \
    --admm_dual_dtype=fp32 \
    --admm_split_dtype=fp32 \
    --save_model=true \
    --admm_save_path=/home1/doyoonkim/projects/elsa/models \
    --eval_math500=false \
    --eval_zero_shot=true \
    --eval_full_bench=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b_nostrip8192 \
    --seed=42 \
    --push_to_hub=true \
    --run_name_suffix="elsaplain_s${SPARSITY_PCT}pct_lr${LR}_lmda${LMDA}_nostrip8192"

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
