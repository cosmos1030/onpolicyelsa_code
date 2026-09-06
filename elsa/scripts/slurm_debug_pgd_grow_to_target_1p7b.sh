#!/bin/bash
#SBATCH --job-name=debug_pgd_grow
#SBATCH --partition=A100-40GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/debug_pgd_grow_%j.out
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61
exec 2>&1

# Smoke test for --gmp_pgd_grow_to_target: PGD-driven growth, no separate
# TR-GMP/cubic/cosine growth mechanism at all -- _pgd_desired targets
# final_sparsity directly and the self-KL budget alone paces how fast dead
# count grows toward it (revive saturates at min(k, revive_cand) instead of
# being forced equal to prune). Cheap/short run just to confirm no crashes
# and that sparsity actually climbs toward target over these few steps
# instead of staying flat (which would mean prune_cand isn't actually
# outrunning revive_cand as designed).
#
# --gmp_pruning_end_ratio=0.0 (with --gmp_tr_enabled=false) disables the
# schedule-driven growth path entirely (pruning_end_steps=0 -> the
# `step <= pruning_end_steps` check that gates maskmgr.update() is false
# from step 1 onward -- see gmp_trainer.py's mask_interval block, `else`
# branch) -- PGD, gated separately by its own step%pgd_interval==0 check,
# is left as the ONLY thing that can move the mask.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"

export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

echo "=== debug: gmp_pgd_grow_to_target smoke test (1.7B, single GPU) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --kd_nsamples=256 \
    --sparsity_ratio=0.7 \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --steps=100 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --lr=1e-4 \
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
    --gmp_tr_enabled=false \
    --gmp_growth_schedule=cosine \
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
    --run_name_suffix="debug_pgd_grow_to_target" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
