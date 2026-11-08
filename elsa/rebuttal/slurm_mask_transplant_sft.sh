#!/bin/bash
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=n3,n19,n31,n42,n43,n46,n51,n52,n54,n55,n58,n60,n61,n64,n76,n77,n80,n84,n87,n90,n91,n92
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1
# Rebuttal mask transplant: dense Qwen3-1.7B * MASK -> fixed-mask SCOUT losses NTP/KD/OPD 0.33 each (roll512, interval 32)
# short recovery (STEPS), identical data/seed across arms; only the mask differs.
# Usage: sbatch --job-name=<name> slurm_mask_transplant_sft.sh <MASK_DIR> <ARM_TAG> [SWAP_FROM_FINAL] [STEPS]
MASK_DIR=${1:?mask}; ARM=${2:?arm}; SWAP_FINAL=${3:-}; STEPS=${4:-64}
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
DENSE="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/slurm" /home1/doyoonkim/projects/elsa/logs
NFS_LOG="/home1/doyoonkim/projects/elsa/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT
export WANDB_DIR="$LOCAL_JOB_BASE/wandb" WANDB_SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=120 TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER} HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== mask transplant SFT arm=${ARM} mask=${MASK_DIR} swap_final=${SWAP_FINAL} steps=${STEPS} ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then echo "ERROR: No internet on $(hostname)."; exit 1; fi
cd /home1/doyoonkim/projects/elsa
INIT=$LOCAL_JOB_BASE/masked_dense_${ARM}
$PYTHON rebuttal/make_masked_dense.py --dense "$DENSE" --mask_dir "$MASK_DIR" ${SWAP_FINAL:+--swap_from_final "$SWAP_FINAL"} --seed 0 --out "$INIT" || exit 1
$PYTHON main.py \
    --model="$INIT" \
    --gmp_teacher_model="$DENSE" \
    --dataset=mixed_cot \
    --data_path=/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    --sparsity_ratio=0.7 \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --steps=${STEPS} \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=1e-4 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=8 \
    --seqlen=8192 \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_kl_chunk_size=256 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=32 \
    --gmp_onpolicy_max_new_tokens=512 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_opkd_vllm_sidecar=true \
    --gmp_prompt_path=/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b_nostrip8192 \
    --run_name_suffix=rb_masktransplant_${ARM}_fixedmask_ntpkdopd_steps${STEPS}_roll512 \
    --seed=42
EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
