#!/bin/bash
#SBATCH --job-name=cubic_ntpkdopkd_4b
#SBATCH --partition=H200-PCIe-ZT
#SBATCH --qos=zt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n89,n90,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/cubic_ntpkdopkd_4b_%j.out
exec 2>&1

# Cubic-schedule (--gmp_tr_enabled=false) counterpart of the best mi=32
# TR-GMP NTP+KD+OPKD Qwen3-4B S70 baseline (wbid ef5mng93: s70, lr=5e-5,
# mi=32, kl=0.02, seqlen=8192, nostrip8192 data, steps=2048, single GPU no
# FSDP) -- for a fair cubic-vs-trust-region comparison, matched so the
# cubic curve reaches final_sparsity at EXACTLY the step the TR baseline
# actually reached it empirically (verified via wandb: step 1056, NOT the
# gmp_sparse_train_steps=512 in that run's config, which was a dead value
# unused by the TR growth path). --gmp_sparse_train_steps=992 here makes
# pruning_end_steps = steps(2048) - 992 = 1056, so the cubic ramp finishes
# exactly there too, then both recipes do the same fixed-mask sparse-
# training tail for the remaining 992 steps.
#
# --gmp_cubic_log_kl=true measures KL(old||candidate) at every mi=32 mask-
# update boundary (same _compute_tr_kl the TR path itself uses to accept/
# reject growth) purely for diagnostic logging (cubic/kl_before_after,
# cubic/sparsity) -- cubic never gates on it, this just quantifies how far
# outside a trust-region KL budget the forced cubic growth actually lands.
#
# Usage: sbatch slurm_gmp_cubic_ntpkdopkd_qwen3_4b.sh <SPARSITY> <LR> [MASK_INTERVAL] [SPARSE_TRAIN_STEPS] [DATA_PATH]
# e.g.: sbatch slurm_gmp_cubic_ntpkdopkd_qwen3_4b.sh 0.7 5e-5 32 992

SPARSITY=${1:?"Usage: <SPARSITY> <LR> [MASK_INTERVAL] [SPARSE_TRAIN_STEPS] [DATA_PATH]"}
LR=${2:?"Usage: <SPARSITY> <LR> [MASK_INTERVAL] [SPARSE_TRAIN_STEPS] [DATA_PATH]"}
MASK_INTERVAL=${3:-32}
SPARSE_TRAIN_STEPS=${4:-992}
DATA_PATH=${5:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Cubic-schedule TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) Qwen3-4B s${SPARSITY_PCT} lr=${LR} mi=${MASK_INTERVAL} sparse_train_steps=${SPARSE_TRAIN_STEPS} (matched to TR baseline ef5mng93's reach-step=1056) ==="
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
    --sparsity_type=unstructured \
    --do_gmp=true \
    --steps=2048 \
    --gmp_sparse_train_steps=${SPARSE_TRAIN_STEPS} \
    --gmp_dense_warmup_steps=0 \
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
    --gmp_onpolicy_max_new_tokens=512 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=false \
    --gmp_cubic_log_kl=true \
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
    --run_name_suffix="cubic_lr${LR}_mi${MASK_INTERVAL}_matchTR" \
    --seed=42

echo "##### END #####"
