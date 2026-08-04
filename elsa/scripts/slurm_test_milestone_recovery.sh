#!/bin/bash
#SBATCH --job-name=test_ms_recovery
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/local-data/user-data/%u/test_ms_recovery_%j/slurm_%j.out
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
exec 2>&1

# Test: milestone checkpoint saves AFTER mask_interval recovery steps
# Short run: Qwen3-0.6B, 300 steps, mask_interval=32, milestone=0.5,0.6,0.7
# Expected: S50 milestone crossed around step ~80 (fast schedule on 300 total steps)
#           checkpoint saved at ~step 112 (80+32), etc.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL=/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca

LOCAL_JOB_BASE="/local-data/user-data/${USER}/test_ms_recovery_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/models"

export WANDB_MODE=disabled
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE
export TMPDIR=/tmp
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}_test

echo "=== Milestone Recovery Test ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path=/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
    --sparsity_ratio=0.7 \
    --do_gmp=true \
    --gmp_use_fsdp=false \
    --gmp_steps=300 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=1 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=32 \
    --gmp_max_seq_len=512 \
    --gmp_max_prompt_len=256 \
    --gmp_ntp_lambda=0.5 \
    --gmp_kd_lambda=0.5 \
    --gmp_milestone_sparsities=0.5,0.6 \
    --gmp_save_path="$LOCAL_JOB_BASE/models" \
    --save_model=true \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_zero_shot=false \
    --eval_full_bench=false \
    --wandb=false \
    2>&1 | tee "$LOCAL_JOB_BASE/train.log"

echo ""
echo "=== Saved milestone checkpoints ==="
ls -la "$LOCAL_JOB_BASE/models/" 2>/dev/null || echo "(no models dir)"

echo ""
echo "=== Milestone log lines ==="
grep -i "milestone\|sparsity" "$LOCAL_JOB_BASE/train.log" | grep -v "^$"

echo "##### END #####"
