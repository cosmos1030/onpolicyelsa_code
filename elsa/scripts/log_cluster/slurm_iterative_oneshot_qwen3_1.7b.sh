#!/bin/bash
#SBATCH --job-name=iter_oneshot_1.7b
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/onpolicyelsa_code/elsa/logs/iter_oneshot_1.7b_%j.out
exec 2>&1

# Rebuttal SWVr (ii)/(iv): iterative SparseGPT / ALPS baseline, Qwen3-1.7B s70.
# Dense start -> one-shot prune the CURRENT student at 8 stages (steps
# 0,32,...,224; cubic ramp to 70%, target reached at 224 vs SCOUT's 248) ->
# fixed-mask NTP+KD+OPD training in between and after, 2048 steps total.
# Everything else is the 1.7B s70 SCOUT rebuttal recipe (wandb ic8vljhs):
# lr 1e-4 cosine, warmup 256, batch 1x8, seqlen 8192, rollouts 512 tok,
# refresh 32, sidecar vLLM, OT80/FW20.
# Calibration per stage: 128 x 2048 windows of the same corpus (paper one-shot
# baselines' calibration), drawn once.
#
# Usage: sbatch slurm_iterative_oneshot_qwen3_1.7b.sh <sparsegpt|alps>
# Debug (short): DEBUG=1 sbatch slurm_iterative_oneshot_qwen3_1.7b.sh sparsegpt

METHOD=${1:?"Usage: sbatch $0 <sparsegpt|alps>"}
SPARSITY=0.7
REPO_ROOT="/home/doyoonkim/onpolicyelsa_code/elsa"
DATA_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa/data"
DATA_PATH="${DATA_ROOT}/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
OPD_PROMPT_PATH="${DATA_ROOT}/ot3_fineweb_200k_qwen3_opdprompts.jsonl"
MODEL="/home/shared/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

if [ "${DEBUG:-0}" = "1" ]; then
    STEPS=40; STAGES="0,32"; STAGE_SPS="0.3,0.5"; NCAL=8
    SAVE=false; PUSH=false; EVAL=false; WANDB=false; CKPT_EVERY=0; TAG="debug"
else
    STEPS=2048; STAGES="${STAGES:-0,32,64,96,128,160,192,224}"; STAGE_SPS=""; NCAL=128
    SAVE=true; PUSH=true; EVAL=true; WANDB=true; CKPT_EVERY=256; TAG="rebuttal"
fi

for f in "$DATA_PATH" "$OPD_PROMPT_PATH" "$MODEL/config.json"; do
    if [ ! -e "$f" ]; then echo "ERROR: missing $f" >&2; exit 1; fi
done

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

LOCAL_JOB_BASE="/tmp/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "${REPO_ROOT}/logs" "${REPO_ROOT}/models"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=${WANDB_API_KEY:-$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)}
export HF_HOME=/home/shared/huggingface
export HF_HUB_DISABLE_XET=1
# in-process vLLM (sleep mode, CuMemAllocator) asserts on expandable_segments
if [ "${SIDECAR:-true}" = "true" ]; then export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; else export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256; fi
export TOKENIZERS_PARALLELISM=false
export ELSA_EVAL_SKIP_TASKS=${ELSA_EVAL_SKIP_TASKS:-gpqa}
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "=== iterative ${METHOD} Qwen3-1.7B s70 stages=${STAGES} (${TAG}) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [ "$WANDB" = "true" ] && ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

if [ "${SIDECAR:-true}" = "true" ]; then SIDECAR_TAG="sidecar_"; else SIDECAR_TAG="inproc_"; fi
[ "${LEGACY_SYNC:-false}" = "true" ] && SIDECAR_TAG="${SIDECAR_TAG}legacysync_"

cd "$REPO_ROOT"

python main.py \
    --model="$MODEL" \
    --gmp_teacher_model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --gmp_tr_enabled=false \
    --gmp_pgd=false \
    --gmp_oneshot_method=${METHOD} \
    --gmp_oneshot_stage_steps="${STAGES}" \
    --gmp_oneshot_stage_sparsities="${STAGE_SPS}" \
    --gmp_oneshot_calib_nsamples=${NCAL} \
    --gmp_oneshot_calib_seqlen=2048 \
    --gmp_oneshot_calib_seed=42 \
    --steps=${STEPS} \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=1e-4 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_mask_interval=32 \
    --seqlen=8192 \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_kl_chunk_size=256 \
    --kd_nsamples=0 \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=32 \
    --gmp_onpolicy_max_new_tokens=512 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_opkd_vllm_sidecar=${SIDECAR:-true} \
    --gmp_opkd_legacy_partial_sync=${LEGACY_SYNC:-false} \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_ckpt_every_steps=${CKPT_EVERY} \
    --gmp_ckpt_dir="${REPO_ROOT}/models/ckpt_iter_${METHOD}_1.7b_s70_${SIDECAR_TAG}${TAG}_${SLURM_JOB_ID}" \
    --gmp_resume_from="${RESUME_FROM:-}" \
    --gmp_save_path="${REPO_ROOT}/models" \
    --save_model=${SAVE} \
    --push_to_hub=${PUSH} \
    --eval_math500=false \
    --eval_full_bench=${EVAL} \
    --eval_profile=long \
    --eval_zero_shot=false \
    --wandb=${WANDB} \
    --wandb_project=reasoning_qwen3_1.7b_nostrip8192 \
    --seed=42 \
    --run_name_suffix="iter_${METHOD}_$(echo "$STAGES" | tr "," "\n" | wc -l)stage_cubic224_lr1e-4_ri32_${SIDECAR_TAG}${TAG}"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
