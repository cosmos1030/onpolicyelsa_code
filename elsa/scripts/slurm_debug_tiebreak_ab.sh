#!/bin/bash
#SBATCH --job-name=debug_tiebreak
#SBATCH --partition=A100-40GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:40:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91,n87,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/debug_tiebreak_%j.out
exec 2>&1

# A/B: _pgd_topk_mask_from_vals's tie-cluster random-thinning block (added
# 5a6cb61, 08-31 22:24) ON (default) vs OFF (GMP_PGD_SKIP_TIEBREAK=1, reverts
# to the old all-or-nothing threshold cut) -- exact same harness as
# slurm_debug_kl_chunk_ab.sh (1.7B, S50, mi=32, ro=32, kl=0.02, lr=5e-5, 48
# steps, no eval/push), chunk_size fixed at 2048 (current default, already
# shown to cost only ~2% in that A/B) -- isolates the tie-breaking block's
# own wall-clock cost as a separate hypothesis for the 10h->13h+ production
# slowdown, since it lands in the SAME candidate code window (de4b58b
# 12:01 -> 5a6cb61 22:24 on 08-31) as gmp_kl_chunk_size and touches every
# PGD bisection call (up to ~7x/step), not just the final application.
#
# Usage: sbatch slurm_debug_tiebreak_ab.sh <on|off>

MODE=${1:?"Usage: <on|off> (on=tie-break enabled/default, off=GMP_PGD_SKIP_TIEBREAK=1)"}
if [ "$MODE" = "off" ]; then
  export GMP_PGD_SKIP_TIEBREAK=1
fi

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
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

echo "=== debug: tie-break A/B mode=${MODE} (GMP_PGD_SKIP_TIEBREAK=${GMP_PGD_SKIP_TIEBREAK:-unset}) (1.7B, S50, mi=32, ro=32, kl=0.02, lr=5e-5, chunk=2048, 48 steps, no eval/push) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --kd_nsamples=2048 \
    --sparsity_ratio=0.5 \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --steps=48 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=5e-5 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=32 \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --gmp_pruning_scope=global \
    --seqlen=8192 \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_kl_chunk_size=2048 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=32 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=0.02 \
    --gmp_tr_kl_reduce=mean \
    --gmp_pgd=true \
    --gmp_pgd_kl_budget=0.02 \
    --gmp_pgd_kl_calib_size=4 \
    --gmp_pgd_interval=1 \
    --gmp_pgd_skip_growth_step=true \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=false \
    --run_name_suffix="debug_tiebreak_ab_${MODE}" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
