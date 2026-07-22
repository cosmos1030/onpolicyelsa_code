#!/bin/bash
#SBATCH --job-name=debug_gmp_fsdp_7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --exclude=n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/debug_gmp_fsdp_7b_%j.out
exec 2>&1

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb

export WANDB_DIR=/local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

echo "=== DEBUG: GMP FSDP 7B ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "NODE: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    main.py \
    --model=$MODEL \
    --dataset=math_cot \
    --data_path=$DATA_PATH \
    --sparsity_ratio=0.5 \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --gmp_steps=20 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=2 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=8 \
    --gmp_fisher_beta=0.999 \
    --gmp_kd_lambda=0.0 \
    --gmp_max_seq_len=512 \
    --gmp_max_prompt_len=256 \
    --save_model=false \
    --eval_math500=false \
    --eval_zero_shot=false \
    --wandb=false \
    --push_to_hub=false \
    --seed=42

echo "=== DEBUG DONE ==="
