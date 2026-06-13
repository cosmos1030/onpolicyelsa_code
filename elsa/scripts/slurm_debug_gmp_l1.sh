#!/bin/bash
#SBATCH --job-name=debug_gmp_l1
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --exclude=n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/debug_gmp_l1_%j.out
exec 2>&1

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb

export WANDB_DIR=/local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1 | tr -d ' \n\r')
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

SPARSITY_TYPE=${1:-"unstructured"}
SPARSITY_RATIO=${2:-"0.7"}
L1_LAMBDA=${3:-"1e-3"}

echo "=== DEBUG: GMP L1 reg (sparsity_type=$SPARSITY_TYPE ratio=$SPARSITY_RATIO l1=$L1_LAMBDA) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

/home1/doyoonkim/miniconda3/envs/rac/bin/python main.py \
    --model=$MODEL \
    --dataset=math_cot \
    --data_path=$DATA_PATH \
    --sparsity_ratio=$SPARSITY_RATIO \
    --sparsity_type=$SPARSITY_TYPE \
    --do_gmp=true \
    --gmp_steps=40 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=8 \
    --gmp_kd_lambda=0.0 \
    --gmp_max_seq_len=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_l1_lambda=$L1_LAMBDA \
    --save_model=false \
    --eval_math500=false \
    --eval_zero_shot=false \
    --wandb=false \
    --push_to_hub=false \
    --seed=42

echo "=== DEBUG DONE ==="
