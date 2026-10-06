#!/bin/bash
#SBATCH --job-name=debug_blockwise_stall
#SBATCH --partition=A100-40GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/debug_blockwise_stall_%j.out
exec 2>&1

# Debug-only smoke test for --gmp_blockwise_squarehead: cheap, single-GPU,
# non-FSDP, NTP+KD only (no OPKD/PGD), short run, tight mask_interval so we
# hit several TR-GMP growth events quickly and can observe block widening
# in the log if/when growth stalls.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

echo "=== debug: gmp_blockwise_squarehead smoke test (1.7B, single GPU) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --kd_nsamples=256 \
    --sparsity_ratio=0.9 \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --steps=40 \
    --gmp_post_target_steps=8 \
    --gmp_sparse_train_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --lr=5e-5 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=8 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=8 \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --gmp_pruning_scope=global \
    --seqlen=1024 \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.5 \
    --gmp_kd_lambda=0.5 \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=0.0005 \
    --gmp_tr_kl_reduce=mean \
    --gmp_blockwise_squarehead=true \
    --gmp_blockwise_hardness=1.0 \
    --gmp_blockwise_init_block=1 \
    --gmp_blockwise_widen_factor=2 \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=false \
    --run_name_suffix="debug_blockwise_stall" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
