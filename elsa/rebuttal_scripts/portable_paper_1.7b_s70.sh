#!/bin/bash
# PORTABLE copy for the other server (handoff 2026-11-08). Paths come from env:
#   ELSA_WT, PYTHON (required); MODEL, DATA_DIR, SAVE_DIR, LOCAL_BASE (optional).
# Submit via rebuttal_scripts/handoff_submit.sh, which also sets partition/qos/output.
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
exec 2>&1
# Rebuttal: re-run of the PAPER 1.7B s70 SCOUT config (wandb q9f33zl7, job
# 823374, HF gmp-kd3e-1-s70pct-lr1e-4_20260901_084730) on the paper's own
# commit 5a6cb61 (worktree /home1/doyoonkim/projects_wt_paper5a6cb61), env rac,
# in-process vLLM (real sync bug), paper-era in-job lighteval.
# Only deltas vs the paper command line:
#   --gmp_kl_chunk_size 2048->256 (same loss value; avoids the 1.24GiB
#     contiguous-block SIGSEGV under max_split_size_mb:256, no resume in this commit)
#   --push_to_hub=false (HF storage full since 11/02), --eval_zero_shot=false
#   + ported --gmp_pgd_freeze_after_target / --gmp_save_reach_mask
# Usage: sbatch --job-name=<name> slurm_paper_1.7b_s70.sh <FREEZE true|false> <RUN_TAG>
FREEZE=${1:?freeze}
RUN_TAG=${2:?tag}
WT=${ELSA_WT:?set ELSA_WT to <this worktree>/elsa}
PYTHON=${PYTHON:?set PYTHON to the env python used for SCOUT runs (vllm 0.x in-process, VLLM_USE_V1=0)}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
LOCAL_JOB_BASE="${LOCAL_BASE:-/tmp/${USER}}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$WT/logs" "$WT/models/reach_masks"
NFS_LOG="$WT/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.local.out"
# stdout already goes to $WT/logs via --output (see handoff_submit.sh); no local copy needed
REACH_MASK=$WT/models/reach_masks/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.pt
export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=120 TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER} HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== PAPER-COMMIT 5a6cb61 1.7B sparsity=${SPARSITY:-0.7} freeze=${FREEZE} tag=${RUN_TAG} extra=[${EXTRA_FLAGS:-}] ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  HEAD=$(git -C $WT rev-parse --short HEAD)"
git -C $WT diff --stat
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then echo "ERROR: No internet on $(hostname)."; exit 1; fi
cd $WT
$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path=${DATA_DIR:-$WT/data}/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    --sparsity_ratio=${SPARSITY:-0.7} --sparsity_type=unstructured --do_gmp=true --steps=2048 \
    --gmp_batch_size=1 --gmp_grad_accum=8 --lr=1e-4 --lr_scheduler=cosine --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 --gmp_mask_interval=32 --gmp_fisher_beta=0.999 --gmp_saliency=fisher \
    --gmp_pruning_scope=global --seqlen=8192 --gmp_gradient_checkpointing=true --gmp_max_prompt_len=512 \
    --gmp_kd_only=false --gmp_kl_chunk_size=256 --kd_nsamples=0 \
    --gmp_ntp_lambda=0.33 --gmp_kd_lambda=0.33 --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=32 --gmp_onpolicy_max_new_tokens=512 --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path=${DATA_DIR:-$WT/data}/ot3_fineweb_200k_qwen3_opdprompts.jsonl \
    --gmp_tr_enabled=false --gmp_pruning_end_ratio=0.0 --gmp_pgd=true --gmp_pgd_grow_to_target=true \
    --gmp_pgd_kl_budget=0.02 --gmp_pgd_kl_calib_size=4 --gmp_pgd_interval=8 \
    --gmp_pgd_freeze_after_target=${FREEZE} --gmp_save_reach_mask="$REACH_MASK" \
    --gmp_save_path=${SAVE_DIR:-$WT/models} --save_model=true --push_to_hub=false \
    --eval_math500=false --eval_full_bench=true --eval_zero_shot=false \
    --wandb=true --wandb_project=reasoning_qwen3_1.7b_nostrip8192 \
    ${EXTRA_FLAGS:-} \
    --run_name_suffix=${RUN_TAG}_pgd_grow2target_klbudget0.02_lr1e-4_pgdi8_globalscope_ot3_fineweb_40k_qwen3_nostrip_8192 \
    --seed=42
EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
