#!/bin/bash
#SBATCH --job-name=dbg_dpo_premask
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --exclude=n80
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/dbg_dpo_premask_%j.out
exec 2>&1

mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

echo "=== DEBUG: pruning-aware DPO (pre-mask ref) ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

/home1/doyoonkim/miniconda3/envs/rac/bin/python main.py \
    --model=$MODEL \
    --dataset=math_cot \
    --data_path=$DATA_PATH \
    --sparsity_ratio=0.7 \
    --do_gmp=true \
    --gmp_steps=10 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=2 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=4 \
    --gmp_kd_lambda=0.0 \
    --gmp_max_seq_len=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_dpo_lambda=0.03 \
    --gmp_dpo_beta=0.1 \
    --gmp_dpo_n_pairs=8 \
    --gmp_dpo_gen_batch=4 \
    --gmp_dpo_max_new_tokens=64 \
    --gmp_dpo_temperature=0.7 \
    --gmp_dpo_start_step=0 \
    --gmp_dpo_reference_free=false \
    --save_model=false \
    --eval_math500=false \
    --eval_zero_shot=false \
    --wandb=false \
    --push_to_hub=false \
    --seed=42

echo "=== DEBUG DONE ==="
