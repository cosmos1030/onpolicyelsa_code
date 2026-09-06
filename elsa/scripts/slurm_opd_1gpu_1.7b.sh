#!/bin/bash
#SBATCH --job-name=opd_1gpu_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=150G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/opd_1gpu_1.7b_%j/slurm_%j.out
exec 2>&1

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

# Hyperparams (override via env vars)
ADMM_LR=${ADMM_LR:-1e-4}
ADMM_LMDA=${ADMM_LMDA:-0.05}

LOCAL_JOB_BASE="/local-data/user-data/${USER}/opd_1gpu_1.7b_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/tmp
export VLLM_USE_V1=0

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "CONFIG: lr=${ADMM_LR} admm_lmda=${ADMM_LMDA}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

# Full-scale OPD: single GPU, 2048 steps, NTP+KD+OPD = 1/3 each (opd_lambda=1.0)
$PYTHON main.py \
    --model="$MODEL" \
    --do_kd_admm=true \
    --data_path="$DATA_PATH" \
    --dataset=mixed_cot \
    --sparsity_ratio=0.5 \
    --admm_steps=2048 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=4 \
    --admm_lmda=${ADMM_LMDA} \
    --admm_lmda_schedule_mode=constant \
    --admm_lr=${ADMM_LR} \
    --admm_base_optimizer=adamw \
    --admm_beta2=0.999 \
    --admm_interval=32 \
    --admm_precision=bf16 \
    --admm_dual_dtype=bf16 \
    --admm_split_dtype=bf16 \
    --admm_tr_z_proj=true \
    --admm_tr_kl_threshold=0.2 \
    --admm_tr_init_delta=0.05 \
    --admm_tr_delta_min=0.001 \
    --admm_tr_max_iters=8 \
    --admm_lasso_lmda=0 \
    --kd_use_cot_dataset=true \
    --kd_offpolicy_ntp=true \
    --kd_forward_kl=true \
    --kd_lambda=0.5 \
    --kd_ntp_lambda=0.5 \
    --kd_max_prompt_len=512 \
    --opd_enabled=true \
    --opd_lambda=1.0 \
    --opd_vllm_max_tokens=256 \
    --opd_vllm_gpu_mem=0.30 \
    --save_model=true \
    --admm_save_path=/home1/doyoonkim/projects/elsa/models \
    --eval_math500=false \
    --eval_zero_shot=false \
    --eval_full_bench=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42 \
    --push_to_hub=true

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
