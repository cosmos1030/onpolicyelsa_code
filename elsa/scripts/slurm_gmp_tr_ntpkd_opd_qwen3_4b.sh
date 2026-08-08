#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opd_4b
#SBATCH --partition=H200
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n87
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_ntpkd_opd_4b_%j.out
exec 2>&1

# General-purpose TR-GMP NTP+KD+OPD (0.33/0.33/0.33) launcher for Qwen3-4B,
# generalized from slurm_gmp_tr_ntpkd_opd_qwen3_4b_olddata.sh (parameterizes
# DATA_PATH/LR/MASK_INTERVAL instead of hardcoding the old-dataset repro).
# Single H200 (141GB) -- 4B + AdamW + vLLM OPD engine together, no FSDP.
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b.sh <SPARSITY> <KL_THRESHOLD> [MASK_INTERVAL] [LR] [DATA_PATH] [OPD_PROMPT_PATH]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b.sh 0.6 0.02 32 5e-5 \
#         /home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_thinkstrip.jsonl \
#         /home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl

SPARSITY=${1:?"Usage: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_4b.sh <SPARSITY> <KL_THRESHOLD> [MASK_INTERVAL] [LR] [DATA_PATH] [OPD_PROMPT_PATH]"}
KL_THRESHOLD=${2:-0.02}
MASK_INTERVAL=${3:-32}
LR=${4:-5e-5}
DATA_PATH=${5:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_thinkstrip.jsonl}
OPD_PROMPT_PATH=${6:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

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

echo "=== TR-GMP NTP+KD+OPD(0.33/0.33/0.33) Qwen3-4B s${SPARSITY_PCT} lr=${LR} kl=${KL_THRESHOLD} mask_interval=${MASK_INTERVAL} data=$(basename $DATA_PATH) ==="
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
    --gmp_post_target_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --seqlen=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_kd_only=false \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_saliency=fisher \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_use_fsdp=false \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b \
    --run_name_suffix="s${SPARSITY_PCT}_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}_$(basename $DATA_PATH .jsonl)" \
    --seed=42

echo "##### END #####"
