#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opd_8b_fsdp2
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/tr_ntpkd_opd_8b_fsdp2_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPD (0.33/0.33/0.33) for Qwen3-8B, 2-GPU FSDP (FULL_SHARD)
# instead of single-GPU: a single-H200 attempt (41450-41455) OOM'd at step 1
# inside the full-vocab KD KL loss (~136GB peak, seqlen=8192 x ~152k vocab).
# FSDP shards student weights/grad/optimizer across both GPUs (teacher stays
# a full unsharded replica per rank -- see main.py's gmp_use_fsdp block), so
# per-GPU fixed footprint drops to roughly teacher(16GB) + student-shard
# (~32GB) = ~48GB, leaving plenty of room for the full-vocab KL intermediate
# without needing gmp_kd_topk/gmp_onpolicy_kd_topk (kept at full-vocab, 0).
#
# vLLM (OPD rollouts) shares GPU0 (gmp_opkd_vllm_gpu_index=0, tp_size=1)
# instead of a dedicated 3rd GPU or a tensor-parallel split across both --
# TP=2 was tried and confirmed to deadlock even as a fully independent
# subprocess.Popen process (not nested in torchrun) the moment ANOTHER live
# torch.distributed pg (this job's own FSDP group) exists on the same
# physical GPUs, reproduced down to a minimal repro with no FSDP/8B model at
# all (scripts/log_cluster/repro_nested_nccl_test.py). Two co-existing,
# uncoordinated distributed runtimes on one GPU just don't work here --
# TRL's vllm_mode="colocate" gets away with it because ITS runtime owns both
# lifecycles and phase-separates generation/training; our sidecar has no such
# coordination layer. GPU0 ends up ~69GB (48GB FSDP shard + 21GB vLLM),
# GPU1 ~48GB -- imbalanced but safely inside 141GB. Revisit tp_size=2 only
# after porting real phase-separated (train/rollout) lifecycle control.
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_8b_fsdp2gpu.sh <SPARSITY> <LR> <KL_THRESHOLD> <DATA_PATH> \
#          [MASK_INTERVAL] [LR_SCHEDULER] [SALIENCY] [WANDB_PROJECT] [EVAL_PROFILE]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_8b_fsdp2gpu.sh 0.5 1e-4 0.02 \
#         /home/doyoonkim/projects/onpolicyelsa_code/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
#         32 cosine fisher reasoning_qwen3_8b_nostrip8192 quick

SPARSITY=${1:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <DATA_PATH> [MASK_INTERVAL] [LR_SCHEDULER] [SALIENCY] [WANDB_PROJECT] [EVAL_PROFILE]"}
LR=${2:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <DATA_PATH> [MASK_INTERVAL] [LR_SCHEDULER] [SALIENCY] [WANDB_PROJECT] [EVAL_PROFILE]"}
KL_THRESHOLD=${3:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <DATA_PATH> [MASK_INTERVAL] [LR_SCHEDULER] [SALIENCY] [WANDB_PROJECT] [EVAL_PROFILE]"}
DATA_PATH=${4:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <DATA_PATH> [MASK_INTERVAL] [LR_SCHEDULER] [SALIENCY] [WANDB_PROJECT] [EVAL_PROFILE]"}
REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
OPD_PROMPT_PATH="${REPO_ROOT}/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"
MASK_INTERVAL=${5:-32}
LR_SCHEDULER=${6:-cosine}
SALIENCY=${7:-fisher}
WANDB_PROJECT=${8:-reasoning_qwen3_8b_nostrip8192}
EVAL_PROFILE=${9:-quick}
MODEL="Qwen/Qwen3-8B"
SEQLEN=8192

DATA_TAG=$(basename "$DATA_PATH" .jsonl | sed -E 's/ot3_?//g; s/fineweb_200k_?//g; s/qwen3_?//g; s/^_+//; s/_+$//; s/__+/_/g')

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac
TORCHRUN=torchrun

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
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== TR-GMP NTP+KD+OPD (0.33/0.33/0.33) ${MODEL} s${SPARSITY} lr=${LR} kl=${KL_THRESHOLD} mi=${MASK_INTERVAL} saliency=${SALIENCY} data=${DATA_TAG} -- 2xH200 FSDP, vLLM sharing GPU0 ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd "$REPO_ROOT"

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
    --gmp_saliency=${SALIENCY} \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_save_path="${REPO_ROOT}/models" \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_profile=${EVAL_PROFILE} \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --seed=42 \
    --run_name_suffix="${SALIENCY}_${DATA_TAG}_fsdp2"

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
