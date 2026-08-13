#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opkd_24_4b
#SBATCH --partition=H200-PCIe-ZT
#SBATCH --qos=zt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n89,n90,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_ntpkd_opkd_24_4b_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) for Qwen3-4B, 2:4 semi-structured
# sparsity, SINGLE A100-80GB (no FSDP) -- the FSDP-4gpu attempt (719626/627)
# crashed in loss.backward() with a tensor-shape mismatch: the structured-L1
# (2:4 grouping) code assumes full unsharded 2D weight tensors, but FSDP
# hands each rank a flat 1D shard, so it's fundamentally incompatible with
# FSDP as currently written. 4B unstructured TR-GMP already runs fine on a
# single A100-80GB (no FSDP needed at all -- only 8B needed FSDP), and the
# 2:4 L1 regularizer is a cheap elementwise op, so single-GPU should have
# plenty of headroom. Same recipe as the validated 1.7B 2:4 canary (718464).
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b.sh <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b.sh 0.5 1e-4 0.02

SPARSITY=${1:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]"}
LR=${2:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]"}
KL_THRESHOLD=${3:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]"}
MASK_INTERVAL=${4:-32}
L1_LAMBDA=${5:-0.0001}
DATA_PATH=${6:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
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
# NOTE: expandable_segments is incompatible with vLLM's CuMemAllocator,
# which the OPKD vLLM engine now requires (enable_sleep_mode=True, added in
# the 2026-08-13 log_cluster pull) -- LLM(...) hard-asserts on this at
# load_model() time if set. Use max_split_size_mb instead -- a different
# fragmentation mitigation the CuMemAllocator assertion doesn't check for
# (it only greps for the literal string "expandable_segments:True") --
# leaving fragmentation completely unmitigated caused a real OOM after
# ~760 steps on the 1.7B single-GPU 2:4 canary (720073).
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) 2:4 Qwen3-4B s${SPARSITY_PCT} lr=${LR} kl=${KL_THRESHOLD} mi=${MASK_INTERVAL} l1=${L1_LAMBDA} (single A100, OT80/FW20) ==="
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
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=2:4 \
    --gmp_l1_lambda=${L1_LAMBDA} \
    --do_gmp=true \
    --steps=2048 \
    --gmp_post_target_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
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
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_use_fsdp=false \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b_nostrip8192 \
    --run_name_suffix="24_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
