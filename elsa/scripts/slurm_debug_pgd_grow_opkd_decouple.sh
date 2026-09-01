#!/bin/bash
#SBATCH --job-name=debug_pgd_grow_opkd_decouple
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/debug_pgd_grow_opkd_decouple_%j.out
exec 2>&1

# Validates the fix that decouples pgd2growth's (--gmp_tr_enabled=false)
# OPKD vLLM rollout-pool refill from --gmp_mask_interval. Before the fix,
# refill was hardcoded to `step % mask_interval == 0` regardless of
# --gmp_onpolicy_kd_interval's value whenever tr_enabled=false; the new
# "pgd2growth OPKD pool refill" block in gmp_trainer.py (~line 4942) now
# gates it on `step % onpolicy_interval == 0` instead, fully independent
# of mask_interval.
#
# Deliberately mismatched on purpose: mask_interval=32 (default),
# onpolicy_interval=8 (4x MORE frequent). Before the fix this would have
# been silently ignored (refill still only at step 32, 64, ...). After the
# fix, expect "OPKD vLLM pool refilled (pgd2growth, onpolicy_interval=8)"
# log lines at step=8,16,24,32,... -- i.e. every 8 steps, NOT gated to the
# mask_interval=32 boundary.
#
# Qwen3-0.6B / single GPU / 40 steps / short seqlen -- cheap, just checking
# the refill cadence + no crash, not real training quality.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1
export NCCL_DEBUG=WARN

echo "=== debug: pgd2growth OPKD pool refill decoupled from mask_interval (Qwen3-0.6B) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --kd_nsamples=256 \
    --sparsity_ratio=0.5 \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --steps=40 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=2 \
    --lr=1e-4 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=32 \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --gmp_pruning_scope=global \
    --seqlen=512 \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=256 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.34 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=8 \
    --gmp_onpolicy_max_new_tokens=64 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=false \
    --gmp_pruning_end_ratio=0.0 \
    --gmp_pgd=true \
    --gmp_pgd_grow_to_target=true \
    --gmp_pgd_kl_budget=0.02 \
    --gmp_pgd_kl_calib_size=4 \
    --gmp_pgd_kl_calib_seqlen=256 \
    --gmp_pgd_kl_bisect_iters=6 \
    --gmp_pgd_interval=1 \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=debug_pgd_convergence \
    --run_name_suffix="debug_pgd_grow_opkd_decouple" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
