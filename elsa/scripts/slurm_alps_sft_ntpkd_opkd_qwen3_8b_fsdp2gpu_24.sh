#!/bin/bash
#SBATCH --job-name=alps_sft_ntpkd_opkd_8b_fsdp2_24
#SBATCH --partition=H200
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n76,n77,n80,n87,n91,n61,n64
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/alps_sft_ntpkd_opkd_8b_fsdp2_24_%j.out
exec 2>&1

# 2:4 semi-structured variant of slurm_alps_sft_ntpkd_opkd_qwen3_8b_fsdp2gpu.sh
# (the PROVEN config -- job 728889 completed fully with results): loads the
# ALPS one-shot 2:4 checkpoint (qwen3_8b_alps_s24, job 719641), freezes its
# mask, same NTP+KD+OPKD(0.33/0.33/0.33) recovery recipe/budget, 2xH200 FSDP
# with vLLM sharing GPU0 (gmp_opkd_vllm_gpu_mem=0.15 -- on H200's 141GB that's
# ~21GB, comfortably fitting 8B's ~16GB bf16 weights + KV cache; the SAME
# 0.15 fraction on an 80GB A100 (only ~12GB) was too small to even load the
# model and crashed vLLM immediately on every attempt regardless of node --
# see the abandoned _fsdp4gpu_24 variant's failure history). Kept at 2 GPUs
# (not 4) specifically to reuse this exact proven config unchanged.
#
# Usage: sbatch slurm_alps_sft_ntpkd_opkd_qwen3_8b_fsdp2gpu_24.sh [LR] [OPD_GEN_LEN] \
#          [LR_SCHEDULER] [DATA_PATH] [SEQLEN] [MASK_INTERVAL] [WANDB_PROJECT]
# e.g.: sbatch slurm_alps_sft_ntpkd_opkd_qwen3_8b_fsdp2gpu_24.sh 1e-4

LR=${1:-1e-4}
OPD_GEN_LEN=${2:-256}
LR_SCHEDULER=${3:-cosine}
DATA_PATH=${4:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${5:-8192}
MASK_INTERVAL=${6:-32}
WANDB_PROJECT=${7:-reasoning_qwen3_8b_nostrip8192}

SPARSITY=0.5
SPARSITY_TYPE=2:4
ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_8b_alps_s24"
SPARSITY_TAG="n24"

DENSE_MODEL=$(ls -d /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ 2>/dev/null | head -1)
DENSE_MODEL="${DENSE_MODEL%/}"
if [ -z "$DENSE_MODEL" ] || [ ! -f "$DENSE_MODEL/config.json" ]; then
    echo "ERROR: Qwen3-8B not found in HF cache" >&2
    exit 1
fi
if [ ! -d "$ALPS_MODEL" ]; then
    echo "ERROR: ALPS checkpoint not found at $ALPS_MODEL -- run ALPS/slurm_alps_prune_8b_rtx6000ada.sh ${SPARSITY} first" >&2
    exit 1
fi

OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"
TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun

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

echo "=== ALPS -> Sparse SFT NTP+KD+OPKD(0.33/0.33/0.33) Qwen3-8B ${SPARSITY_TAG} (2:4 semi-structured) lr=${LR} opd_gen_len=${OPD_GEN_LEN} seqlen=${SEQLEN} -- 2xH200 FSDP, vLLM sharing GPU0 ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$ALPS_MODEL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=2 --master_port=${MASTER_PORT} main.py \
    --model="$ALPS_MODEL" \
    --gmp_teacher_model="$DENSE_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=${SPARSITY_TYPE} \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --gmp_use_fsdp=true \
    --steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_kd_only=false \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_opkd_vllm_gpu_index=0 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_profile=quick \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --seed=42 \
    --run_name_suffix="alpssft_${SPARSITY_TAG}_lr${LR}_$(basename "$DATA_PATH" .jsonl)_fsdp2"

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
