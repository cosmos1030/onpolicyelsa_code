#!/bin/bash
#SBATCH --job-name=elsa_trz_ntp_s70_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/elsa_trz_ntp_s70_1.7b_%j/slurm_%j.out
exec 2>&1

# Same base config as wandb run c2pp4ud8 (ntp_admm_s0.7_lr0.0001_lmda0.005_steps8192,
# reasoning_qwen3_1.7b), plus the fixed TR-z (prev-z vs candidate-z KL trust region,
# lib/trainer.py:_compute_kl_with_z + lib/optimizers.py z0-init fix).
# Usage: sbatch slurm_elsa_trz_ntp_s70_lr1e4_1.7b.sh <KL_THRESHOLD>

KL_THRESHOLD=${1:?"Usage: sbatch slurm_elsa_trz_ntp_s70_lr1e4_1.7b.sh <KL_THRESHOLD>"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/elsa_trz_ntp_s70_1.7b_${SLURM_JOB_ID}"
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

echo "=== TR-z NTP-ADMM Qwen3-1.7B s70 kl=${KL_THRESHOLD} ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --data_path="$DATA_PATH" \
    --dataset=math_cot \
    --sparsity_ratio=0.7 \
    --admm_steps=2048 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=8 \
    --admm_lmda=0.005 \
    --admm_init_lmda=0 \
    --admm_final_lmda=0.005 \
    --admm_lmda_schedule_mode=cosine \
    --admm_lr=1e-4 \
    --admm_base_optimizer=adamw \
    --admm_beta1=0.9 \
    --admm_beta2=0.999 \
    --admm_interval=32 \
    --admm_precision=bf16 \
    --admm_dual_dtype=fp32 \
    --admm_split_dtype=fp32 \
    --admm_tr_z_proj=true \
    --admm_tr_kl_threshold=${KL_THRESHOLD} \
    --admm_tr_init_delta=0.05 \
    --admm_tr_delta_min=0.001 \
    --admm_tr_max_iters=8 \
    --save_model=true \
    --admm_save_path=/home1/doyoonkim/projects/elsa/models \
    --eval_math500=false \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42 \
    --push_to_hub=true

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
