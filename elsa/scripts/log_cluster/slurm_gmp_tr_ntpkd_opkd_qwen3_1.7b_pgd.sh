#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opkd_1.7b_pgd
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/tr_ntpkd_opkd_1.7b_pgd_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) for Qwen3-1.7B with --gmp_pgd=true --
# log_cluster port of the other server's
# rerun_ot80fw20/slurm_gmp_tr_ntpkd_opkd_qwen3_1.7b_pgd.sh (same recipe
# behind that server's 707782-793 baseline + PGD, first real exercise of
# this flag there). After each optimizer step the mask is fully
# re-projected from fresh fisher-importance scores (top-k binary search at
# fixed sparsity) instead of only ever growing monotonically -- params can
# be pruned even if kept before ("prunings") and revived even if pruned
# before ("revivals"). Single H200, no FSDP -- 1.7B is tiny, plenty of
# headroom even with the OPKD vLLM sidecar sharing the GPU.
#
# GMP_BATCH_SIZE default here is 2 (not 1, unlike the 4B/8B scripts) --
# global batch stays 8 via GMP_GRAD_ACCUM=4. Untested at this batch size on
# this model/recipe combination -- watch the first few steps for OOM; if it
# doesn't fit, drop to --gmp_batch_size=1 --gmp_grad_accum=8 (this script's
# own earlier default, and what every other single-GPU TR-GMP script here
# uses).
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opkd_qwen3_1.7b_pgd.sh <SPARSITY> <KL_THRESHOLD> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [POST_TARGET_STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [SPARSITY_TYPE] [L1_LAMBDA] [BATCH_SIZE] [GRAD_ACCUM]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opkd_qwen3_1.7b_pgd.sh 0.5 0.02 512 32 cosine 2048 0 1e-4 \
#         /home/doyoonkim/projects/onpolicyelsa_code/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
#         8192 true reasoning_qwen3_1.7b_nostrip8192 fisher global 0.33,0.33,0.33 2:4 0.0001

SPARSITY=${1:?"Usage: <SPARSITY> <KL_THRESHOLD> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [POST_TARGET_STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [SPARSITY_TYPE] [L1_LAMBDA] [BATCH_SIZE] [GRAD_ACCUM]"}
KL_THRESHOLD=${2:-0.01}
OPD_GEN_LEN=${3:-256}
MASK_INTERVAL=${4:-8}
LR_SCHEDULER=${5:-cosine}
STEPS=${6:-2048}
POST_TARGET_STEPS=${7:-8}
LR=${8:-1e-4}
REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
DATA_PATH=${9:-${REPO_ROOT}/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${10:-8192}
GRAD_CKPT=${11:-true}
WANDB_PROJECT=${12:-reasoning_qwen3_1.7b_nostrip8192}
SALIENCY=${13:-fisher}
PRUNING_SCOPE=${14:-global}
LOSS_WEIGHTS=${15:-0.33,0.33,0.33}  # NTP,KD,OPKD
SPARSITY_TYPE=${16:-unstructured}   # unstructured | 2:4 | 4:8
L1_LAMBDA=${17:-0.0}
BATCH_SIZE=${18:-2}
GRAD_ACCUM=${19:-4}
NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

MODEL="Qwen/Qwen3-1.7B"
OPD_PROMPT_PATH="${REPO_ROOT}/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

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
export HF_HUB_DISABLE_XET=1
# NOT expandable_segments:True -- OPKD's vLLM engine runs with
# enable_sleep_mode=True (single-GPU path), whose CuMemAllocator hard-asserts
# expandable_segments is unset at load_model() time. Use max_split_size_mb
# instead -- doesn't trip that assertion but still mitigates fragmentation
# (real OOM precedent: job 720073 on the other server's 1.7B 2:4 canary).
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "=== TR-GMP NTP+KD+OPKD(${LOSS_WEIGHTS}) PGD ${MODEL} s${SPARSITY_PCT} (${SPARSITY_TYPE}) lr=${LR} kl=${KL_THRESHOLD} mi=${MASK_INTERVAL} batch=${BATCH_SIZE}x${GRAD_ACCUM} saliency=${SALIENCY} (single H200) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd "$REPO_ROOT"

python main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=${SPARSITY_TYPE} \
    --gmp_l1_lambda=${L1_LAMBDA} \
    --do_gmp=true \
    --steps=${STEPS} \
    --gmp_post_target_steps=${POST_TARGET_STEPS} \
    --gmp_batch_size=${BATCH_SIZE} \
    --gmp_grad_accum=${GRAD_ACCUM} \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=${SALIENCY} \
    --gmp_pruning_scope=${PRUNING_SCOPE} \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=${GRAD_CKPT} \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=${KD_ONLY} \
    --gmp_ntp_lambda=${NTP_LAMBDA} \
    --gmp_kd_lambda=${KD_LAMBDA} \
    --gmp_onpolicy_kd_lambda=${OPKD_LAMBDA} \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_pgd=true \
    --gmp_use_fsdp=false \
    --gmp_save_path="${REPO_ROOT}/models" \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_profile=quick \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --run_name_suffix="pgd_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}_${PRUNING_SCOPE}scope_b${BATCH_SIZE}x${GRAD_ACCUM}" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
