#!/bin/bash
#SBATCH --job-name=dbg_tropkd_2a100_shvllm
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/dbg_tropkd_2a100_shvllm_%j.out
exec 2>&1

# Debug: verify vLLM subprocess can share GPU 0 with FSDP rank0 (gmp_opkd_vllm_gpu_index=0)
# instead of needing a dedicated 3rd GPU. 2x A100-80GB, 20 steps only, no save/push/eval.
# If this doesn't OOM, the real 4B ef job can run on 2 GPUs total (no idle vLLM-only GPU).

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_20k.jsonl"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== DEBUG: TR-GMP OPKD(Dense) Qwen3-4B 2xA100-80GB, vLLM sharing GPU0 ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MASTER_PORT=$MASTER_PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

mkdir -p /home1/doyoonkim/projects/elsa/logs
cd /home1/doyoonkim/projects/elsa

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=${MASTER_PORT} \
    main.py \
    --model="$MODEL" \
    --dataset=math_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=0.7 \
    --do_gmp=true \
    --gmp_steps=20 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.0 \
    --gmp_mask_interval=4 \
    --gmp_fisher_beta=0.999 \
    --gmp_max_seq_len=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_opkd_vllm_gpu_index=0 \
    --gmp_prompt_path="$DATA_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.005 \
    --gmp_tr_kl_threshold=0.1 \
    --gmp_tr_kl_reduce=mean \
    --gmp_fisher_source=opd_empirical \
    --gmp_use_fsdp=true \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=false \
    --seed=42

echo "=== EXIT: $? ==="
