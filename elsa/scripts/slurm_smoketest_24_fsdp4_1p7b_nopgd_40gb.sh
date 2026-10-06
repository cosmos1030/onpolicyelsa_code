#!/bin/bash
#SBATCH --job-name=smoketest_24_fsdp4_1p7b_nopgd
#SBATCH --partition=A100-40GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/smoketest_24_fsdp4_1p7b_nopgd_%j.out
exec 2>&1

# One-off smoke test: PGD + 2:4 (N:M) + structured-L1 (lasso) + FSDP (4x
# A100-40GB), 40 steps, save_model=true so the checkpoint can be inspected
# for real 2:4 structural validity afterward. Exercises the newly-added
# FSDP gather/reconstruct/scatter shim for both growth (candidate_masks)
# and PGD (_pgd_nm_pre_target/_pgd_nm_post_target) N:M logic -- classic
# FSDP1 shards a Linear weight into an arbitrary flat byte-range chunk per
# rank (verified empirically, not row/col-aligned, can even be empty on a
# rank), so without this fix N:M growth AND PGD would both silently no-op
# under FSDP (dim<2 fallback) instead of crashing -- worse than a crash,
# since it looks like it's running fine.
# global batch = gmp_batch_size(1) * gmp_grad_accum(2) * world_size(4) = 8.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"

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

echo "=== smoketest: PGD + 2:4 + lasso + FSDP4 (A100-40GB) ==="
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
    --gmp_pgd=false \
    --save_model=true \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=false \
    --run_name_suffix="smoketest_24_fsdp4_1p7b_nopgd" \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
