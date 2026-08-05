#!/bin/bash
#SBATCH --job-name=tr_kd_opkd_4b_3gpu
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_kd_opkd_4b_3gpu_%j.out
exec 2>&1

# Same as slurm_gmp_tr_kd_opkd_qwen3_4b.sh (TR-GMP KD+OPKD only, no NTP,
# Fisher saliency, kd_lambda=0.5/onpolicy_kd_lambda=0.5, disjoint KD-train
# vs OPD-prompt splits) but on 3xA100-80GB instead of 1xH200 (H200 queue
# was too congested -- only 2 nodes total, exclude list halves that to 1).
#
# GPU layout: 2 training ranks (FSDP, gmp_use_fsdp=true) + 1 dedicated vLLM
# GPU for OPD (index = world_size = 2, isolated CUDA context -- see
# main.py's _use_fsdp_opkd). global batch = gmp_batch_size(1) *
# gmp_grad_accum(4) * world_size(2) = 8.
#
# Usage: sbatch slurm_gmp_tr_kd_opkd_qwen3_4b_fsdp3gpu.sh <SPARSITY> <KL_THRESHOLD>
# e.g.: sbatch slurm_gmp_tr_kd_opkd_qwen3_4b_fsdp3gpu.sh 0.6 0.01

SPARSITY=${1:?"Usage: sbatch slurm_gmp_tr_kd_opkd_qwen3_4b_fsdp3gpu.sh <SPARSITY> <KL_THRESHOLD>"}
KL_THRESHOLD=${2:-0.01}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_train.jsonl"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

DEBUG_LOG_COPY="/home1/doyoonkim/projects/elsa/logs/tr_kd_opkd_4b_3gpu_${SLURM_JOB_ID}_last.out"
copy_log_on_exit() { cp "/home1/doyoonkim/projects/elsa/logs/tr_kd_opkd_4b_3gpu_${SLURM_JOB_ID}.out" "$DEBUG_LOG_COPY" 2>/dev/null || true; }
trap copy_log_on_exit EXIT

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

echo "=== TR-GMP KD+OPKD(0.5/0.5, no NTP) Qwen3-4B s${SPARSITY_PCT} kl=${KL_THRESHOLD}, 2xA100 FSDP train + 1xA100 dedicated-vLLM (OT80/FW20) ==="
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
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=32 \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --seqlen=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=true \
    --gmp_kd_lambda=0.5 \
    --gmp_onpolicy_kd_lambda=0.5 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.25 \
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
    --wandb_project=reasoning_qwen3_4b \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
