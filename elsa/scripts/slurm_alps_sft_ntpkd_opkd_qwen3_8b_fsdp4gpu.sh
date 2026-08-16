#!/bin/bash
#SBATCH --job-name=alps_sft_ntpkd_opkd_8b_fsdp4
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n76,n77,n80,n87,n91,n61,n64
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/alps_sft_ntpkd_opkd_8b_fsdp4_%j.out
exec 2>&1

# Same as slurm_alps_sft_ntpkd_opkd_qwen3_8b_fsdp2gpu.sh (ALPS one-shot pruned
# -> fixed-mask NTP+KD+OPKD(0.33/0.33/0.33) recovery training, Qwen3-8B,
# matching the TR-GMP NTP+KD+OPKD 8B sweep for a budget-matched comparison)
# but on 4xA100-80GB instead of 2xH200 -- the H200 partition (only 2 nodes,
# 8 GPUs each) is fully occupied by other jobs, while A100-80GB has plenty of
# free capacity. Sharding the student across 4 GPUs instead of 2 leaves each
# GPU with roughly half the memory footprint of the 2-GPU version, which
# should give enough headroom for the vLLM OPD-rollout slice to coexist on a
# 80GB card even though the 2-GPU version needed 141GB H200s -- same trick
# already validated for TR-GMP 4B 2:4 on 4xA100
# (slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_fsdp4gpu.sh): vLLM colocates on
# training rank 0's physical GPU (gmp_opkd_vllm_gpu_index=0, gpu_mem=0.15)
# instead of a dedicated extra GPU, and TP=1 (not TP=2) avoids the NCCL
# deadlock between vLLM's process group and FSDP's own.
#
# grad_accum lowered from 8 (2-GPU version) to 4 to keep the effective global
# batch size the same (1*4*4=16 here vs 1*8*2=16 there), for a fair
# comparison against both the 2-GPU ALPS-SFT run and the TR-GMP FSDP2 sweep.
#
# Usage: sbatch slurm_alps_sft_ntpkd_opkd_qwen3_8b_fsdp4gpu.sh <SPARSITY> [LR] [OPD_GEN_LEN] \
#          [LR_SCHEDULER] [DATA_PATH] [SEQLEN] [MASK_INTERVAL] [WANDB_PROJECT]
# e.g.: sbatch slurm_alps_sft_ntpkd_opkd_qwen3_8b_fsdp4gpu.sh 0.5 1e-4

SPARSITY=${1:?"Usage: <SPARSITY> [LR] [OPD_GEN_LEN] [LR_SCHEDULER] [DATA_PATH] [SEQLEN] [MASK_INTERVAL] [WANDB_PROJECT]"}
LR=${2:-1e-4}
OPD_GEN_LEN=${3:-256}
LR_SCHEDULER=${4:-cosine}
DATA_PATH=${5:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${6:-8192}
MASK_INTERVAL=${7:-32}
WANDB_PROJECT=${8:-reasoning_qwen3_8b_nostrip8192}

SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_8b_alps_s${SPARSITY_PCT}pct"
SPARSITY_TAG="s${SPARSITY_PCT}pct"

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

echo "=== ALPS -> Sparse SFT NTP+KD+OPKD(0.33/0.33/0.33) Qwen3-8B ${SPARSITY_TAG} lr=${LR} opd_gen_len=${OPD_GEN_LEN} seqlen=${SEQLEN} -- 4xA100-80GB FSDP, vLLM sharing GPU0 ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$ALPS_MODEL"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=4 --master_port=${MASTER_PORT} main.py \
    --model="$ALPS_MODEL" \
    --gmp_teacher_model="$DENSE_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --gmp_use_fsdp=true \
    --steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
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
    --gmp_opkd_vllm_gpu_mem=0.3 \
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
    --run_name_suffix="alpssft_${SPARSITY_TAG}_lr${LR}_$(basename "$DATA_PATH" .jsonl)_fsdp4"

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
