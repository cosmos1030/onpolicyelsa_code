#!/bin/bash
#SBATCH --job-name=debug_blockwise_scope_delay_stall
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/debug_blockwise_scope_delay_stall_%j.out
exec 2>&1

# Smoke test for the new combo: --gmp_pruning_scope=block (Fisher importance
# computed/ranked SEPARATELY within each block-of-layers group instead of one
# global threshold) + --gmp_blockwise_delay_global_signal=true (NTP/KD held
# at 0 -- SquareHead loss alone drives training and Fisher importance --
# until block_size widens all the way to every layer, then switches back on).
# OPKD left off (lambda=0) to keep this cheap; only testing that block-scope
# pruning + the lambda-delay/reactivation plumbing don't crash.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"

export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

echo "=== debug: gmp_pruning_scope=block + gmp_blockwise_delay_global_signal smoke test (1.7B, single GPU) ==="
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
    --steps=60 \
    --gmp_sparse_train_steps=0 \
    --gmp_post_target_steps=8 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --lr=1e-4 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=8 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=8 \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --gmp_pruning_scope=block \
    --seqlen=1024 \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.5 \
    --gmp_kd_lambda=0.5 \
    --gmp_onpolicy_kd_lambda=0 \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=0.0005 \
    --gmp_tr_kl_reduce=mean \
    --gmp_blockwise_squarehead=true \
    --gmp_blockwise_hardness=1.0 \
    --gmp_blockwise_init_block=1 \
    --gmp_blockwise_widen_factor=2 \
    --gmp_blockwise_delay_global_signal=true \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=false \
    --run_name_suffix="debug_blockwise_scope_delay_stall" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
