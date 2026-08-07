#!/bin/bash
#SBATCH --job-name=gmp7b_s05_lr5e-5
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --exclude=n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/gmp7b_s05_lr5e-5_%j.out
exec 2>&1
mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb

export WANDB_DIR=/local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

echo "=== Job gmp7b_s05_lr5e-5 | sp=0.5 lr=5e-5 ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "NODE: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    main.py \
    --model=/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60 \
    --dataset=mixed_cot \
    --data_path=/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl \
    --sparsity_ratio=0.5 \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --gmp_steps=1024 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=2 \
    --gmp_lr=5e-5 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=32 \
    --gmp_fisher_beta=0.999 \
    --gmp_kd_lambda=0.0 \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --gmp_max_prompt_len=512 \
    --gmp_max_seq_len=2048 \
    --save_model=true \
    --eval_math500=true \
    --math500_max_new_tokens=8192 \
    --math500_max_samples=500 \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=gmp_qwen3_7b \
    --push_to_hub=false \
    --seed=42
