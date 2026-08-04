#!/bin/bash
#SBATCH --job-name=fsdp_test_0.6b
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=0-00:20:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/fsdp_test_3090_%j.out
exec 2>&1

mkdir -p /home1/doyoonkim/projects/elsa/logs

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
echo "NODE=$(hostname) PORT=$MASTER_PORT GPUs=4"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=${MASTER_PORT} \
    main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=0.5 \
    --admm_steps=128 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=4 \
    --admm_lmda=0.1 \
    --admm_lmda_schedule_mode=constant \
    --admm_lr=1e-5 \
    --admm_base_optimizer=adamw \
    --admm_beta2=0.999 \
    --admm_interval=32 \
    --admm_precision=bf16 \
    --admm_dual_dtype=bf16 \
    --admm_split_dtype=bf16 \
    --admm_use_fsdp=true \
    --admm_tr_z_proj=true \
    --admm_tr_kl_threshold=0.5 \
    --admm_tr_init_delta=0.05 \
    --admm_tr_delta_min=0.001 \
    --admm_tr_max_iters=8 \
    --kd_max_prompt_len=512 \
    --save_model=false \
    --eval_math500=false \
    --eval_zero_shot=false \
    --wandb=false \
    --kd_nsamples=256 \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
