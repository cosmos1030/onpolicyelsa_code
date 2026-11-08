#!/bin/bash
#SBATCH --job-name=alpssft8b_fsdp2
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/onpolicyelsa_code/elsa/logs/alpssft8b_fsdp2_%j.out
exec 2>&1

# ALPS one-shot -> fixed-mask NTP+KD+OPD (0.33/0.33/0.33) recovery training,
# Qwen3-8B, log-master H200, 2-GPU FSDP.
#
# Rebuttal (Q4RB Q-Q2): learning-rate sweep for the Figure 3 ALPS+training
# baseline at 8B s70 (paper value lr=1e-4, wandb 3zpks6yu).
# Same recipe as the paper's Table 1 and as the 8B SCOUT run (ajm5l60w):
#   global batch 8 = batch 1 x grad_accum 4 x 2 ranks (ajm5l60w layout),
#   rollout pool 256 = 32 x 4 x 2, rollout refresh every 32 steps,
#   rollout length 512, 2048 steps, cosine, warmup 256, seqlen 8192.
# FSDP path syncs vLLM via load_weights (not affected by the in-process
# qkv/gate_up sync bug fixed in 856f814).
#
# ALPS start: cosmos1030/alps-n128-qwen3-8b-s70pct (standard ALPS, same
# data / nsamples 128 / rho 300 / seed 42 as the paper checkpoint
# qwen3-8b-alps-s70pct, which is no longer on the Hub).
#
# Usage: sbatch slurm_alps_sft_ntpkd_opkd_qwen3_8b_fsdp2_h200.sh <LR> [SPARSITY] [ALPS_MODEL_DIR]
# e.g.:  sbatch slurm_alps_sft_ntpkd_opkd_qwen3_8b_fsdp2_h200.sh 5e-5

LR=${1:?"Usage: sbatch $0 <LR> [SPARSITY] [ALPS_MODEL_DIR]"}
SPARSITY=${2:-0.7}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
SPARSITY_TAG="s${SPARSITY_PCT}pct"

REPO_ROOT="/home/doyoonkim/onpolicyelsa_code/elsa"
DATA_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa/data"
DATA_PATH="${DATA_ROOT}/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
OPD_PROMPT_PATH="${DATA_ROOT}/ot3_fineweb_200k_qwen3_opdprompts.jsonl"
ALPS_MODEL=${3:-${REPO_ROOT}/models/qwen3_8b_alps_n128_s${SPARSITY_PCT}pct}
DENSE_MODEL="/home/shared/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

for f in "$DATA_PATH" "$OPD_PROMPT_PATH" "$ALPS_MODEL/config.json" "$DENSE_MODEL/config.json"; do
    if [ ! -e "$f" ]; then echo "ERROR: missing $f" >&2; exit 1; fi
done

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

LOCAL_JOB_BASE="/tmp/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "${REPO_ROOT}/logs" "${REPO_ROOT}/models"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=${WANDB_API_KEY:-$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)}
export HF_HOME=/home/shared/huggingface
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export ELSA_EVAL_SKIP_TASKS=${ELSA_EVAL_SKIP_TASKS:-gpqa}
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== ALPS -> NTP+KD+OPD Qwen3-8B ${SPARSITY_TAG} lr=${LR} -- 2xH200 FSDP, vLLM on GPU0 ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$ALPS_MODEL  COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd "$REPO_ROOT"

torchrun --nproc_per_node=2 --master_port=${MASTER_PORT} main.py \
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
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --seqlen=8192 \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=${NTP_LAMBDA:-0.33} \
    --gmp_kd_lambda=${KD_LAMBDA:-0.33} \
    --gmp_onpolicy_kd_lambda=${OPKD_LAMBDA:-0.33} \
    --gmp_kd_only=false \
    --gmp_onpolicy_max_new_tokens=512 \
    --gmp_onpolicy_kd_interval=32 \
    --gmp_mask_interval=32 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_opkd_vllm_gpu_index=0 \
    --gmp_opkd_legacy_partial_sync=${LEGACY_SYNC:-false} \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_ckpt_every_steps=${CKPT_EVERY:-0} \
    --gmp_ckpt_dir="${REPO_ROOT}/models/ckpt_alpssft8b_${SPARSITY_TAG}_lr${LR}${TAG_SUFFIX:-}_fsdp2" \
    --gmp_resume_from="${RESUME_FROM:-}" \
    --gmp_save_path="${REPO_ROOT}/models" \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_profile=long \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_8b_nostrip8192 \
    --seed=42 \
    --run_name_suffix="alpssft8b_${SPARSITY_TAG}_lr${LR}${TAG_SUFFIX:-}_rebuttal_lrsweep_h200fsdp2"

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
