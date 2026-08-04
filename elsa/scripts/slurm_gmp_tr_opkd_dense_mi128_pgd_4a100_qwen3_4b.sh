#!/bin/bash
#SBATCH --job-name=tr_opkd_pgd_4a100
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_opkd_pgd_4a100_%j.out
exec 2>&1

# TR-GMP NTP+KD+OPKD (Dense teacher) + PGD, Qwen3-4B, 4x A100-80GB FSDP
# FSDP fix: vLLM init on rank 0 only (dist env vars temporarily removed),
#           pool broadcast via dist.broadcast_object_list to all ranks,
#           weight sync via FSDP.summon_full_params collective.
# Usage: sbatch slurm_gmp_tr_opkd_dense_mi128_pgd_4a100_qwen3_4b.sh <KL_THRESHOLD>

KL_THRESHOLD=${1:?"Usage: sbatch slurm_gmp_tr_opkd_dense_mi128_pgd_4a100_qwen3_4b.sh <KL_THRESHOLD>"}

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

echo "=== TR-GMP NTP+KD+OPKD(Dense)+PGD Qwen3-4B 4xA100 kl=${KL_THRESHOLD} ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MASTER_PORT=$MASTER_PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$TORCHRUN \
    --nproc_per_node=4 \
    --master_port=${MASTER_PORT} \
    main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=0.7 \
    --do_gmp=true \
    --gmp_steps=32768 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --gmp_lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=128 \
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
    --gmp_use_fsdp=true \
    --gmp_pgd=true \
    --gmp_milestone_sparsities="0.5,0.6,0.7" \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b \
    --seed=42
TORCHRUN_EXIT=$?
echo "=== torchrun exit code: $TORCHRUN_EXIT ==="

# === rundb: register milestone results ===
_WBID=$(cat "$WANDB_RUN_ID_OUTPUT" 2>/dev/null | tr -d '\n')
if [ -n "$_WBID" ]; then
    cd /home1/doyoonkim/projects/elsa/scripts
    $PYTHON rundb/cli.py register \
        --model qwen3_4b \
        --sparsities "0.5,0.6,0.7" \
        --badge tropkd \
        --name "TR-GMP NTP+KD + OPKD (Dense) + PGD" \
        --sub "kl=${KL_THRESHOLD}_mi128_pgd_4a100" \
        --wbid "$_WBID" 2>&1 || echo "rundb register failed (non-fatal)"
fi

# === git push results_db.json ===
_GIT_ROOT="/home1/doyoonkim/projects"
git -C "$_GIT_ROOT" add elsa/scripts/results_db.json
if ! git -C "$_GIT_ROOT" diff --cached --quiet; then
    git -C "$_GIT_ROOT" commit -m "chore: auto-update results_db (job ${SLURM_JOB_ID})" \
        && git -C "$_GIT_ROOT" push 2>&1 \
        || echo "WARNING: git push failed (non-fatal)"
fi

echo "##### END #####"
