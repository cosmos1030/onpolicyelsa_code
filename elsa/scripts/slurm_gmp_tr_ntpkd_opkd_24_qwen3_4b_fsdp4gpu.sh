#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opkd_24_4b_fsdp4
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_ntpkd_opkd_24_4b_fsdp4_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) for Qwen3-4B, 2:4 semi-structured
# sparsity, 4-GPU FSDP (FULL_SHARD) on 4xA100-80GB -- same colocate pattern
# as the newly-pulled 8B/2xH200 script (log_cluster/slurm_gmp_tr_ntpkd_opd_qwen3_8b_fsdp2gpu.sh):
# vLLM (OPKD rollouts) shares training rank 0's physical GPU
# (gmp_opkd_vllm_gpu_index=0) instead of a dedicated extra GPU, since a
# second co-existing NCCL process group on a DIFFERENT GPU works fine, only
# tensor-parallel vLLM sharing GPUs with an active FSDP group deadlocked.
#
# Structured-L1 pre-conditioning (gmp_l1_lambda, bottom-2-per-group,
# gmp_l1_structured=true default) shrinks to-be-pruned weights toward zero
# during normal KL-bounded training steps so the eventual hard 2:4 cut is
# close to a no-op -- see the _structured_l1_loss bug fix (closed groups no
# longer get penalized) validated by canary job 718464 (1xA100, no FSDP).
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_fsdp4gpu.sh <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH] [WANDB_PROJECT]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_fsdp4gpu.sh 0.5 1e-4 0.02

SPARSITY=${1:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH] [WANDB_PROJECT]"}
LR=${2:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH] [WANDB_PROJECT]"}
KL_THRESHOLD=${3:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH] [WANDB_PROJECT]"}
MASK_INTERVAL=${4:-32}
L1_LAMBDA=${5:-0.0001}
DATA_PATH=${6:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
WANDB_PROJECT=${7:-reasoning_qwen3_4b_nostrip8192}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) 2:4 Qwen3-4B s${SPARSITY_PCT} lr=${LR} kl=${KL_THRESHOLD} mi=${MASK_INTERVAL} l1=${L1_LAMBDA} -- 2xA100 FSDP, vLLM sharing GPU0 (OT80/FW20) ==="
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
    --sparsity_type=2:4 \
    --gmp_l1_lambda=${L1_LAMBDA} \
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
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_opkd_vllm_gpu_index=0 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
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
    --wandb_project=${WANDB_PROJECT} \
    --run_name_suffix="24_fsdp4_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}" \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
