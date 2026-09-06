#!/bin/bash
#SBATCH --job-name=tr_opkd_4b_ef_4a100
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_opkd_4b_ef_4a100_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPKD (Dense teacher) Qwen3-4B with empirical Fisher saliency, 3x A100-80GB FSDP
# Backup queue in case H200 stays pending. Same global batch + step count as the single-GPU
# H200 version (slurm_gmp_tr_opkd_dense_qwen3_4b_ef.sh), so training should be equivalent:
#   - H200 (1 GPU):        batch=1 * grad_accum=8 * world_size=1 = global batch 8
#   - here (2 GPU FSDP):    batch=1 * grad_accum=4 * world_size=2 = global batch 8
# DistributedSampler shards data across ranks and FSDP averages grads (gmp_trainer.py:1460),
# so grad_accum must shrink by world_size to keep the global batch identical.
# 2 GPUs for FSDP training (ranks 0-1), 3rd GPU (index 2) dedicated to the vLLM subprocess
# (main.py launches vLLM on device index=world_size — must not overlap training ranks).
# vllm_gpu_mem kept at 0.30 (not raised to 0.80): the vLLM reservation gets NCCL P2P-mapped
# into the training GPUs' address space regardless of physical isolation (see comment in
# main.py ~line 155), and a larger reservation blew up rank0's GPU0 with a 63GB phantom
# allocation (job 675111/675112 OOM). 0.30 matches the previously-stable mi128_pgd_4a100 4b run.
# Usage: sbatch slurm_gmp_tr_opkd_dense_qwen3_4b_ef_4a100.sh <KL_THRESHOLD>

KL_THRESHOLD=${1:?"Usage: sbatch slurm_gmp_tr_opkd_dense_qwen3_4b_ef_4a100.sh <KL_THRESHOLD>"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN

# Free port for torchrun rendezvous
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== TR-GMP NTP+KD+OPKD(Dense) Qwen3-4B Empirical Fisher kl=${KL_THRESHOLD} 4xA100 (milestones: s50/s60/s70) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MASTER_PORT=$MASTER_PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$TORCHRUN \
    --nproc_per_node=2 \
    --master_port=${MASTER_PORT} \
    main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=0.7 \
    --do_gmp=true \
    --gmp_steps=32768 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=32 \
    --gmp_fisher_beta=0.999 \
    --gmp_max_seq_len=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.30 \
    --gmp_prompt_path="$DATA_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.005 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_fisher_source=opd_empirical \
    --gmp_use_fsdp=true \
    --gmp_milestone_sparsities="0.5,0.6,0.7" \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b \
    --seed=42
TORCHRUN_EXIT=$?
echo "=== torchrun exit code: $TORCHRUN_EXIT ==="
[ $TORCHRUN_EXIT -ne 0 ] && { echo "FATAL: main.py failed, skipping rundb/git"; exit $TORCHRUN_EXIT; }

# === rundb: register milestone results ===
_WBID=$(cat "$WANDB_RUN_ID_OUTPUT" 2>/dev/null | tr -d '\n')
if [ -n "$_WBID" ]; then
    cd /home1/doyoonkim/projects/elsa/scripts
    $PYTHON rundb/cli.py register \
        --model qwen3_4b \
        --sparsities "0.5,0.6,0.7" \
        --badge tropkd_ef \
        --name "TR-GMP NTP+KD + OPKD (Dense, Empirical Fisher)" \
        --sub "kl=${KL_THRESHOLD}_4a100" \
        --wbid "$_WBID" 2>&1 || echo "rundb register failed (non-fatal)"
else
    echo "WARNING: wandb run ID not found, skipping rundb register"
fi
# ==========================================

# === git push results_db.json ===
_GIT_ROOT="/home1/doyoonkim/projects"
git -C "$_GIT_ROOT" add elsa/scripts/results_db.json
if ! git -C "$_GIT_ROOT" diff --cached --quiet; then
    git -C "$_GIT_ROOT" commit -m "chore: auto-update results_db (job ${SLURM_JOB_ID})" \
        && git -C "$_GIT_ROOT" push 2>&1 \
        || echo "WARNING: git push failed (non-fatal)"
fi
# ================================

echo "##### END #####"
