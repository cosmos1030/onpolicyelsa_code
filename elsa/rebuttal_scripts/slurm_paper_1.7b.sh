#!/bin/bash
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n19,n31,n42,n43,n46,n51,n52,n54,n55,n58,n60,n61,n64,n76,n77,n80,n84,n87,n90,n91,n92
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1
# Rebuttal: paper-condition 1.7B runs on commit 5068c4c (paper 1.7B s80 SCOUT =
# wandb hz4kw1sr, job on n55, in-process vLLM).
# commit 5068c4c (worktree /home1/doyoonkim/projects_wt_paper5068c4c), env rac,
# in-process vLLM (real sync bug), paper-era in-job lighteval.
# Only deltas vs the paper command line:
#   --gmp_kl_chunk_size 2048->256 (same loss value; avoids the 1.24GiB
#     contiguous-block SIGSEGV under max_split_size_mb:256, no resume in this commit)
#   --push_to_hub=false (HF storage full since 11/02), --eval_zero_shot=false
#   no ported flags (FREEZE arg unused here). Paper s80 SCOUT = wandb hz4kw1sr on this commit.
# Usage: sbatch --job-name=<name> slurm_paper_1.7b_s70.sh <FREEZE true|false> <RUN_TAG>
FREEZE=${1:?freeze}
RUN_TAG=${2:?tag}
WT=/home1/doyoonkim/projects_wt_paper5068c4c/elsa
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/slurm" /home1/doyoonkim/projects/elsa/logs
NFS_LOG="/home1/doyoonkim/projects/elsa/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT
REACH_MASK=/home1/doyoonkim/projects/elsa/models/reach_masks/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.pt
export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=120 TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER} HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== PAPER-COMMIT 5068c4c 1.7B sparsity=${SPARSITY:-0.7} freeze=${FREEZE} tag=${RUN_TAG} extra=[${EXTRA_FLAGS:-}] ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  HEAD=$(git -C $WT rev-parse --short HEAD)"
git -C $WT diff --stat
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then echo "ERROR: No internet on $(hostname)."; exit 1; fi
cd $WT
# In-job retry: the 5a6cb61/5068c4c in-process path dies at random in loss.backward()
# (SIGSEGV, exit 139; see project_backward_segfault_flaky). Same flags/seed/allocator,
# restarted from scratch on the same allocation; stop once the final model was saved.
ATTEMPT_LOG=$(mktemp)
for ATTEMPT in 1 2 3 4; do
  echo "=== attempt $ATTEMPT ==="
  $PYTHON main.py \
      --model="$MODEL" \
      --dataset=mixed_cot \
      --data_path=/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
      --sparsity_ratio=${SPARSITY:-0.7} --sparsity_type=unstructured --do_gmp=true --steps=2048 \
      --gmp_batch_size=1 --gmp_grad_accum=8 --lr=1e-4 --lr_scheduler=cosine --lr_warmup_steps=256 \
      --gmp_warmup_ratio=0.05 --gmp_mask_interval=32 --gmp_fisher_beta=0.999 --gmp_saliency=fisher \
      --gmp_pruning_scope=global --seqlen=8192 --gmp_gradient_checkpointing=true --gmp_max_prompt_len=512 \
      --gmp_kd_only=false --gmp_kl_chunk_size=256 --kd_nsamples=0 \
      --gmp_ntp_lambda=0.33 --gmp_kd_lambda=0.33 --gmp_onpolicy_kd_lambda=0.33 \
      --gmp_onpolicy_kd_interval=32 --gmp_onpolicy_max_new_tokens=512 --gmp_opkd_prev_mask_teacher=false \
      --gmp_opkd_vllm_gpu_mem=0.15 \
      --gmp_prompt_path=/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl \
      --gmp_tr_enabled=false --gmp_pruning_end_ratio=0.0 --gmp_pgd=true --gmp_pgd_grow_to_target=true \
      --gmp_pgd_kl_budget=0.02 --gmp_pgd_kl_calib_size=4 --gmp_pgd_interval=8 \
      --gmp_pgd_jump_to_target=false --gmp_opkd_vllm_sidecar=false \
      --gmp_save_path=/home1/doyoonkim/projects/elsa/models --save_model=true --push_to_hub=false \
      --eval_math500=false --eval_full_bench=true --eval_zero_shot=false \
      --wandb=true --wandb_project=reasoning_qwen3_1.7b_nostrip8192 \
      ${EXTRA_FLAGS:-} \
      --run_name_suffix=${RUN_TAG}_pgd_grow2target_klbudget0.02_lr1e-4_pgdi8_ri32_globalscope_ot3_fineweb_40k_qwen3_nostrip_8192 \
      --seed=42 2>&1 | tee "$ATTEMPT_LOG"
  EXIT_CODE=${PIPESTATUS[0]}
  if [ $EXIT_CODE -eq 0 ] || grep -qa "Saved pruned model" "$ATTEMPT_LOG"; then break; fi
  if [ $EXIT_CODE -ne 139 ]; then echo "non-segfault failure ($EXIT_CODE), not retrying"; break; fi
  echo "=== SIGSEGV on attempt $ATTEMPT (last: $(grep -a "Step [0-9]*/" "$ATTEMPT_LOG" | tail -1 | grep -o "Step [0-9]*/[0-9]*")), retrying ==="
  sleep 20
done
echo "=== main.py EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
