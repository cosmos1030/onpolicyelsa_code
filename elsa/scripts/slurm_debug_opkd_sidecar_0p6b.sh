#!/bin/bash
#SBATCH --job-name=debug_opkd_sidecar
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:50:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# Smoke test for --gmp_opkd_vllm_sidecar (added 2026-09-06): single-GPU OPKD
# with vLLM in an independent OS process SHARING the training GPU, instead of
# the in-process vllm.LLM(enable_sleep_mode=True) whose CuMemAllocator has been
# traced to mid-training SIGSEGV in loss.backward().
#
# What it must show:
#   1. "Launching vLLM sidecar on the shared training GPU (... sleep_mode=True)"
#      then "vLLM sidecar ready", from main.py -- NOT gmp_trainer.py's
#      "OPKD vLLM: initializing engine (single-GPU ...)" in-process message.
#   2. "OPKD vLLM: using pre-built out-of-process engine (sidecar); sleep
#      support=True" -- the adapter really exposes sleep/wake_up.
#   3. Rollouts actually generated over the socket: "OPKD vLLM: initial pool
#      filled with N rollouts", then per-window "pool refilled" lines. A
#      sleep/wake bug surfaces here as a hang or a sleep_error/wake_error.
#   4. 40 steps complete with NO "Segmentation fault" and no
#      "Trying to free a pointer not allocated here".
#
# onpolicy_kd_interval=8 with steps=40 deliberately forces ~5 wake/generate/
# sleep cycles: the socket sleep/wake round-trip is the new code, so it should
# be exercised repeatedly rather than once.
#
# Qwen3-0.6B / seqlen=512 / kd_nsamples=256 / no eval -- mechanics only.

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

# Copy logs to NFS once on exit -- /local-data vanishes with the job. The
# sidecar writes its OWN log to $TMPDIR (see vllm_proc.launch_vllm_server);
# grab it too, since a sidecar-side crash leaves nothing in the trainer log
# beyond "server process died".
NFS_LOG="/home1/doyoonkim/projects/elsa/logs/debug_opkd_sidecar_${SLURM_JOB_ID}_last.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/debug_opkd_sidecar_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true;
      for L in /tmp/vllm_server_*.log; do [ -f "$L" ] && cp "$L" "/home1/doyoonkim/projects/elsa/logs/debug_opkd_sidecar_${SLURM_JOB_ID}_$(basename $L)" 2>/dev/null; done; true' EXIT

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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1

echo "=== debug: --gmp_opkd_vllm_sidecar (Qwen3-0.6B, s70, 40 steps, shared-GPU sidecar) ==="
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
    --gmp_onpolicy_kd_interval=8 \
    --gmp_onpolicy_max_new_tokens=64 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_opkd_vllm_sidecar=true \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=false \
    --gmp_pruning_end_ratio=0.0 \
    --gmp_pgd=true \
    --gmp_pgd_grow_to_target=true \
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
    --run_name_suffix="debug_opkd_sidecar_s70_0p6b" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
