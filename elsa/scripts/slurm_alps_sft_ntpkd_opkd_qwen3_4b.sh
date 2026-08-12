#!/bin/bash
#SBATCH --job-name=alps_sft_ntpkd_opkd_4b
#SBATCH --partition=H200-PCIe-ZT
#SBATCH --qos=zt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n89,n90,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/alps_sft_ntpkd_opkd_4b_%j.out
exec 2>&1

# Same as slurm_alps_sft_ntpkd_opkd_qwen3_1.7b.sh but for Qwen3-4B: loads the
# already ALPS-pruned checkpoint (qwen3_4b_alps_s{50,60,70}pct, see
# ALPS/slurm_alps_prune_4b.sh), freezes its mask (gmp_fixed_mask=true), and
# recovery-trains with the same NTP+KD+OPKD(0.33/0.33/0.33) recipe/data/steps
# as the 4B TR-GMP sweep (log_cluster jobs 41299-41349), for direct
# comparison -- mirrors the 1.7B ALPS->SFT experiment (710115-126).
#
# Usage: sbatch slurm_alps_sft_ntpkd_opkd_qwen3_4b.sh <SPARSITY> [LR] [OPD_GEN_LEN] [LR_SCHEDULER] [DATA_PATH] [SEQLEN] [WANDB_PROJECT]
# e.g.: sbatch slurm_alps_sft_ntpkd_opkd_qwen3_4b.sh 0.5 1e-4

SPARSITY=${1:?"Usage: sbatch slurm_alps_sft_ntpkd_opkd_qwen3_4b.sh <SPARSITY> [LR] [OPD_GEN_LEN] [LR_SCHEDULER] [DATA_PATH] [SEQLEN] [WANDB_PROJECT]"}
LR=${2:-1e-4}
OPD_GEN_LEN=${3:-512}
LR_SCHEDULER=${4:-cosine}
WANDB_PROJECT=${7:-reasoning_qwen3_4b_nostrip8192}

SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_s${SPARSITY_PCT}pct"
SPARSITY_TAG="s${SPARSITY_PCT}pct"

# KD/OPKD teacher must be the ORIGINAL DENSE model, not the pruned student
# checkpoint (main.py `_teacher_model_path` respects --gmp_teacher_model).
DENSE_MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
DATA_PATH="${5:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}"
SEQLEN="${6:-8192}"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

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

echo "=== ALPS -> Sparse SFT NTP+KD+OPKD(0.33/0.33/0.33) Qwen3-4B ${SPARSITY_TAG} lr=${LR} opd_gen_len=${OPD_GEN_LEN} lr_scheduler=${LR_SCHEDULER} seqlen=${SEQLEN} (OT80/FW20 nostrip8192) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$ALPS_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# No-internet nodes (e.g. n89/n90) can't reach api.wandb.ai -- instead of
# exiting, fall back to wandb offline mode so the GPU isn't wasted. Offline
# runs must live on NFS (not /local-data) since local-data becomes
# unreachable once the job's node access is revoked -- `wandb sync` this dir
# later from a node with internet to upload the run.
if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "No internet on $(hostname) -- falling back to WANDB_MODE=offline (sync later)."
    OFFLINE_WANDB_DIR="/home1/doyoonkim/projects/elsa/logs/wandb_offline/job_${SLURM_JOB_ID}"
    mkdir -p "$OFFLINE_WANDB_DIR"
    export WANDB_DIR="$OFFLINE_WANDB_DIR"
    export WANDB_MODE=offline
    echo "  offline wandb dir: $OFFLINE_WANDB_DIR (run: wandb sync $OFFLINE_WANDB_DIR/wandb/offline-run-* later)"
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$ALPS_MODEL" \
    --gmp_teacher_model="$DENSE_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_use_fsdp=false \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --run_name_suffix="alpssft_${SPARSITY_TAG}_lr${LR}_$(basename "$DATA_PATH" .jsonl)" \
    --seed=42

echo "##### END #####"
