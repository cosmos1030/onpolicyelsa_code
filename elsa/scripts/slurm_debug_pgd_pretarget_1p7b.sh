#!/bin/bash
#SBATCH --job-name=debug_pgd_pretarget_1p7b
#SBATCH --partition=A100-40GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/debug_pgd_pretarget_1p7b_%j.out
exec 2>&1

# Debug-only: reproduce the TR-GMP(0.05->0.2 growth)+PGD-pretarget interaction
# small/cheap (1.7B, 4xA100-40GB, 40 steps) with the new [DEBUG PGD-pretarget]
# instrumentation, to see the exact k_actual/n_eligible/n_revive_cand/
# sp_before/sp_after numbers behind the observed one-step 0.2->0.462 jump in
# the real 4B production run (733616).

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== debug: TR-GMP+PGD-pretarget interaction, 2:4+lasso+FSDP4 (1.7B) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=4 --master_port=${MASTER_PORT} main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --kd_nsamples=256 \
    --sparsity_ratio=0.5 \
    --sparsity_type=2:4 \
    --gmp_l1_lambda=0.0001 \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --steps=40 \
    --gmp_post_target_steps=8 \
    --gmp_sparse_train_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=2 \
    --lr=5e-5 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=8 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=8 \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --gmp_pruning_scope=global \
    --seqlen=2048 \
    --gmp_gradient_checkpointing=true \
    --gmp_kl_chunk_size=512 \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.5 \
    --gmp_kd_lambda=0.5 \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=0.02 \
    --gmp_tr_kl_reduce=mean \
    --gmp_pgd=true \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=false \
    --run_name_suffix="debug_pgd_pretarget_1p7b" \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
