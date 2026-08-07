#!/bin/bash
#SBATCH --job-name=elsa_plain_1.7b_kdonly_cosinez_opd_2to4
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/elsa_plain_1.7b_kdonly_cosinez_opd_2to4_%j/slurm_%j.out
exec 2>&1

# Same as slurm_elsa_plain_qwen3_1.7b_kdonly_cosinez_opd.sh (KD(0.5)+OPD(0.5),
# no NTP, disjoint KD-train vs OPD-prompt splits, 2xA100 = 1 train + 1
# dedicated vLLM, cosine-z growth) but growing toward a structured 2:4 mask
# instead of a plain unstructured ratio. Uses the new saliency-based N:M
# growing mechanism (lib/trainer.py _select_z_nm / _find_global_threshold_nm
# / _make_z_from_threshold_nm): saliency is compared GLOBALLY across the
# whole model (same cross-layer reallocation as unstructured growth), with
# prunings capped at prune_m - prune_n = 2 per 4-column block so the
# structural budget is never exceeded mid-schedule; as cur_sp -> 0.5 (=2/4)
# the mask converges to an exact 2:4 pattern (also hard-enforced separately
# by ADMMOptimizer.final_projection at save time). Verified via a toy-tensor
# unit test (cap always respected, exact 2:4 at the target ratio) and a
# 256-step KD-only smoke test (699154: schedule value matched the cosine
# formula exactly, no crash) before this full run.
#
# Usage: sbatch slurm_elsa_plain_qwen3_1.7b_kdonly_cosinez_opd_2to4.sh <LMDA> [COSINE_STEPS]
# e.g.: sbatch slurm_elsa_plain_qwen3_1.7b_kdonly_cosinez_opd_2to4.sh 1e-3

LMDA=${1:?"Usage: sbatch slurm_elsa_plain_qwen3_1.7b_kdonly_cosinez_opd_2to4.sh <LMDA> [COSINE_STEPS]"}
COSINE_STEPS=${2:-1024}

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_train.jsonl"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/elsa_plain_1.7b_kdonly_cosinez_opd_2to4_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

DEBUG_LOG_COPY="/home1/doyoonkim/projects/elsa/logs/elsa_plain_1.7b_kdonly_cosinez_opd_2to4_${SLURM_JOB_ID}_last.out"
mkdir -p /home1/doyoonkim/projects/elsa/logs
copy_log_on_exit() { cp "$LOCAL_JOB_BASE/slurm_${SLURM_JOB_ID}.out" "$DEBUG_LOG_COPY" 2>/dev/null || true; }
trap copy_log_on_exit EXIT

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export WANDB_INIT_TIMEOUT=120
export WANDB_RUN_ID_OUTPUT="/home1/doyoonkim/projects/elsa/logs/handoff_${SLURM_JOB_ID}_wandb_run_id.txt"
export MODEL_PATH_OUTPUT="/home1/doyoonkim/projects/elsa/logs/handoff_${SLURM_JOB_ID}_model_path.txt"
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TMPDIR=/tmp
export NCCL_DEBUG=WARN
export VLLM_HOST_IP=127.0.0.1

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== ELSA KD-only(0.5)+cosine-z(0->2:4 over ${COSINE_STEPS} steps)+OPD(0.5) Qwen3-1.7B, lmda=${LMDA}, 1xA100 train + 1xA100 dedicated-vLLM (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=1 --master_port=${MASTER_PORT} main.py \
    --model="$MODEL" \
    --data_path="$DATA_PATH" \
    --dataset=mixed_cot \
    --sparsity_ratio=0.5 \
    --sparsity_type=2:4 \
    --steps=2048 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=8 \
    --admm_use_fsdp=false \
    --admm_lmda=${LMDA} \
    --admm_lmda_schedule_mode=cosine \
    --admm_tr_z_proj=true \
    --admm_z_schedule_mode=cosine \
    --admm_cubic_steps=${COSINE_STEPS} \
    --lr=1e-4 \
    --lr_scheduler=linear \
    --lr_warmup_steps=0 \
    --seqlen=2048 \
    --admm_base_optimizer=adamw \
    --admm_beta1=0.9 \
    --admm_beta2=0.999 \
    --admm_projection_mode=momentum \
    --admm_interval=32 \
    --admm_precision=bf16 \
    --admm_dual_dtype=fp32 \
    --admm_split_dtype=fp32 \
    --do_offpolicy_kd_admm=true \
    --kd_lambda=0.5 \
    --kd_ntp_lambda=0.0 \
    --kd_topk=0 \
    --kd_use_vllm=false \
    --kd_max_prompt_len=512 \
    --opd_enabled=true \
    --opd_lambda=0.5 \
    --opd_prompt_path="$OPD_PROMPT_PATH" \
    --opd_vllm_max_tokens=256 \
    --opd_vllm_gpu_mem=0.25 \
    --save_model=true \
    --admm_save_path=/home1/doyoonkim/projects/elsa/models \
    --eval_math500=false \
    --eval_zero_shot=true \
    --eval_full_bench=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --seed=42 \
    --push_to_hub=true

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
