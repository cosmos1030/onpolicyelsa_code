#!/bin/bash
#SBATCH --job-name=debug_pgd_jump
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:40:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# Smoke test for --gmp_pgd_jump_to_target (added 2026-09-06). Never-exercised
# code path, so this runs before any production 1.7B launch.
#
# What it must show:
#   1. "PGD one-shot jump ENABLED" at startup (flag validation passed).
#   2. At step 8 (= gmp_pgd_interval, the FIRST PGD projection):
#        [pgd_jump_to_target] ONE-SHOT: accepting all <N> prune candidates
#      and the very next [pgd_kl_budget] line reporting k_actual == that same
#      n_prune_cand, with kl_at(k_actual) almost certainly FAR above the 0.02
#      budget (that overshoot is the point of the ablation, not a bug).
#   3. train/sparsity jumping 0 -> 0.7000 in that one step, NOT creeping. The
#      gradual path this is contrasted against took 464 steps to reach target
#      on 1.7B S70 (kl_budget=0.01) and 272 on 4B S70 (kl_budget=0.02).
#   4. Steps 16, 24, 32, ... : normal KL-gated maintenance resumes -- k_actual
#      back down to small tie-noise values, sparsity staying pinned at 0.7000
#      (no oscillation, no drift).
#
# Qwen3-0.6B / single GPU / 40 steps / seqlen=512 / kd_nsamples=256 / no eval
# -- checking mask mechanics only, not training quality. Mirrors
# slurm_debug_pgd_grow_opkd_decouple.sh's shape (same model//steps/seqlen).
#
# ROLLOUT_INTERVAL is deliberately left at the mask_interval default here: the
# dense-rollout-once arm (onpolicy_kd_interval > steps) is an independent knob
# already validated at 4B/8B, and mixing it in would make a jump-specific
# failure harder to localize.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/slurm"
mkdir -p /home1/doyoonkim/projects/elsa/logs

# Copy the log to NFS once on exit (success or crash) -- /local-data is gone
# the moment the job ends, taking any crash trace with it.
NFS_LOG="/home1/doyoonkim/projects/elsa/logs/debug_pgd_jump_${SLURM_JOB_ID}_last.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/debug_pgd_jump_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# expandable_segments is incompatible with vLLM's CuMemAllocator
# (enable_sleep_mode=True), which the OPKD engine requires -- use
# max_split_size_mb instead.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1

echo "=== debug: --gmp_pgd_jump_to_target one-shot jump (Qwen3-0.6B, s70, 40 steps) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --kd_nsamples=256 \
    --sparsity_ratio=0.7 \
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
    --gmp_kl_chunk_size=2048 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.34 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=32 \
    --gmp_onpolicy_max_new_tokens=64 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=false \
    --gmp_pruning_end_ratio=0.0 \
    --gmp_pgd=true \
    --gmp_pgd_grow_to_target=true \
    --gmp_pgd_jump_to_target=true \
    --gmp_pgd_kl_budget=0.02 \
    --gmp_pgd_kl_calib_size=4 \
    --gmp_pgd_kl_calib_seqlen=256 \
    --gmp_pgd_kl_bisect_iters=6 \
    --gmp_pgd_interval=8 \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=debug_pgd_convergence \
    --run_name_suffix="debug_pgd_jump_to_target_s70_0p6b" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
