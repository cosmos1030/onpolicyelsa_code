#!/bin/bash
#SBATCH --job-name=debug_pgd_klgate_tr
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/debug_pgd_klgate_tr_%j.out
exec 2>&1

# Combined smoke test: REAL TR-GMP adaptive growth (gmp_tr_enabled=true, KL-
# gated delta bisection every mask_interval) + the new per-step self-KL-
# gated PGD reprojection (gmp_pgd_kl_budget). This is the actual agreed
# design (TR-GMP handles the 32-step growth events, PGD's new KL gate
# handles per-step reprojection in between) -- the earlier cosine-schedule
# smoke test only isolated the PGD side for debugging.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"

export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

echo "=== debug: gmp_pgd_kl_budget + gmp_tr_enabled=true smoke test (1.7B, single GPU) ==="
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
    --steps=60 \
    --gmp_post_target_steps=8 \
    --gmp_sparse_train_steps=0 \
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
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=0.02 \
    --gmp_tr_kl_reduce=mean \
    --gmp_pgd=true \
    --gmp_pgd_kl_share=true \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=false \
    --run_name_suffix="debug_pgd_klgate_tr" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
