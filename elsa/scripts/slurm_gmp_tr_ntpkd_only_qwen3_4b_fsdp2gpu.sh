#!/bin/bash
#SBATCH --job-name=tr_ntpkd_only_4b_fsdp2
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_ntpkd_only_4b_fsdp2_%j.out
exec 2>&1

# OPD ablation of the best mi=32 TR-GMP NTP+KD+OPKD Qwen3-4B S70 baseline
# (wbid ef5mng93: s70, lr=5e-5, mi=32, kl=0.02, seqlen=8192, nostrip8192
# data) -- identical config, --gmp_onpolicy_kd_lambda=0 (OPKD dropped,
# NTP/KD renormalized to 0.5/0.5, no OPKD vLLM engine spun up at all).
# 2x A100-80GB FSDP instead of single H200 (unstructured sparsity, no
# structured-L1/N:M grouping code involved -- unlike the 2:4 FSDP attempt
# that broke on full-2D-tensor assumptions, this is fine under FSDP shards).
# global batch = gmp_batch_size(1) * gmp_grad_accum(4) * world_size(2) = 8.
#
# Usage: sbatch slurm_gmp_tr_ntpkd_only_qwen3_4b_fsdp2gpu.sh <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [DATA_PATH]
# e.g.: sbatch slurm_gmp_tr_ntpkd_only_qwen3_4b_fsdp2gpu.sh 0.7 5e-5 0.02 32

SPARSITY=${1:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [DATA_PATH]"}
LR=${2:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [DATA_PATH]"}
KL_THRESHOLD=${3:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [DATA_PATH]"}
MASK_INTERVAL=${4:-32}
DATA_PATH=${5:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
SEQLEN=8192

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
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== TR-GMP NTP+KD only (no OPD, ablation of wbid ef5mng93) Qwen3-4B s${SPARSITY_PCT} lr=${LR} kl=${KL_THRESHOLD} mi=${MASK_INTERVAL}, 2xA100-80GB FSDP, global_batch=8 (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=2 --master_port=${MASTER_PORT} main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --steps=2048 \
    --gmp_post_target_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --lr=${LR} \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.5 \
    --gmp_kd_lambda=0.5 \
    --gmp_onpolicy_kd_lambda=0 \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b_nostrip8192 \
    --run_name_suffix="ntpkdonly_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}_fsdp2" \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
