#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opd_8b_fsdp2_multi
#SBATCH --partition=H200
#SBATCH --qos=normal
#SBATCH --gres=gpu:H200:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=600G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/tr_ntpkd_opd_8b_fsdp2_multi_%j.out
exec 2>&1

# Deterministic-NVLink multi-config launcher: requests the WHOLE 8xH200 node
# in a single SLURM job instead of N separate 2-GPU jobs, so WE assign
# CUDA_VISIBLE_DEVICES ourselves to the 4 known-good same-NVLink-domain
# pairs (0,1)(2,3)(4,5)(6,7) instead of hoping SLURM's gres scheduler
# (topology-blind) lands each 2-GPU job inside one NVLink domain -- see the
# log-node07 NVLink topology note in project memory
# (memory/project_log_cluster.md) and scripts/log_cluster/sbatch_nvlink_aware.sh
# for the retry-based alternative this replaces for the common "N 2-GPU
# configs, one 8-GPU node" case.
#
# Runs up to 4 configs concurrently per batch (one per NVLink pair); if more
# than 4 configs are given, later ones wait for the current batch to fully
# finish (no partial overlap, no risk of a 5th job's CUDA_VISIBLE_DEVICES
# colliding with a still-running one) before starting the next batch.
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_8b_fsdp2gpu_multi.sh \
#          <KL_THRESHOLD> <DATA_PATH> <MASK_INTERVAL> <LR_SCHEDULER> <SALIENCY> <WANDB_PROJECT> <EVAL_PROFILE> \
#          -- <SPARSITY1,LR1> <SPARSITY2,LR2> ...
# e.g.: sbatch slurm_gmp_tr_ntpkd_opd_qwen3_8b_fsdp2gpu_multi.sh 0.02 \
#         /home/doyoonkim/projects/onpolicyelsa_code/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
#         32 cosine fisher reasoning_qwen3_8b_nostrip8192 quick \
#         -- 0.5,1e-4 0.5,5e-5 0.6,1e-4 0.6,5e-5 0.7,1e-4 0.7,5e-5

KL_THRESHOLD=${1:?"Usage: <KL_THRESHOLD> <DATA_PATH> <MASK_INTERVAL> <LR_SCHEDULER> <SALIENCY> <WANDB_PROJECT> <EVAL_PROFILE> -- <SPARSITY,LR>..."}
DATA_PATH=${2:?"Usage: <KL_THRESHOLD> <DATA_PATH> <MASK_INTERVAL> <LR_SCHEDULER> <SALIENCY> <WANDB_PROJECT> <EVAL_PROFILE> -- <SPARSITY,LR>..."}
MASK_INTERVAL=${3:-32}
LR_SCHEDULER=${4:-cosine}
SALIENCY=${5:-fisher}
WANDB_PROJECT=${6:-reasoning_qwen3_8b_nostrip8192}
EVAL_PROFILE=${7:-quick}
shift 7 || true
if [[ "${1:-}" != "--" ]]; then
    echo "ERROR: expected a '--' separator before the SPARSITY,LR config list" >&2
    exit 1
fi
shift
CONFIGS=("$@")
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
    echo "ERROR: no SPARSITY,LR configs given after --" >&2
    exit 1
fi

REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
OPD_PROMPT_PATH="${REPO_ROOT}/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"
MODEL="Qwen/Qwen3-8B"
SEQLEN=8192
DATA_TAG=$(basename "$DATA_PATH" .jsonl | sed -E 's/ot3_?//g; s/fineweb_200k_?//g; s/qwen3_?//g; s/^_+//; s/_+$//; s/__+/_/g')

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/models"
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_HOME=/home/shared/huggingface
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export NCCL_DEBUG=WARN
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  configs=${CONFIGS[*]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd "$REPO_ROOT"

# Same-NVLink-domain pairs on log-node07 (GPUs 0-3 are one NV6 domain, 4-7
# the other; cross-domain is SYS/no NVLink -- see project memory).
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")

run_one() {
    local sparsity=$1 lr=$2 gpus=$3 slot=$4
    local sp_tag lr_tag local_base log_path master_port
    sp_tag=$(python3 -c "print(int(${sparsity}*100))")
    lr_tag=$(echo "$lr" | tr -d '.')
    local_base="/tmp/${USER}/job_${SLURM_JOB_ID}_slot${slot}"
    log_path="${REPO_ROOT}/logs/tr_ntpkd_opd_8b_fsdp2_multi_${SLURM_JOB_ID}_s${sp_tag}_lr${lr_tag}.out"
    mkdir -p "$local_base/wandb"
    master_port=$((29500 + slot % 4))

    echo "=== [slot $slot] sparsity=${sparsity} lr=${lr} GPUs=${gpus} port=${master_port} log=${log_path} ==="

    (
        export CUDA_VISIBLE_DEVICES="$gpus"
        export WANDB_DIR="$local_base/wandb"
        export WANDB_RUN_ID_OUTPUT="$local_base/wandb_run_id"
        export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}_slot${slot}"
        torchrun --nproc_per_node=2 --master_port=${master_port} main.py \
            --model="$MODEL" \
            --dataset=mixed_cot \
            --data_path="$DATA_PATH" \
            --sparsity_ratio=${sparsity} \
            --sparsity_type=unstructured \
            --do_gmp=true \
            --gmp_use_fsdp=true \
            --steps=2048 \
            --gmp_post_target_steps=0 \
            --gmp_batch_size=1 \
            --gmp_grad_accum=8 \
            --lr=${lr} \
            --lr_scheduler=${LR_SCHEDULER} \
            --lr_warmup_steps=256 \
            --seqlen=${SEQLEN} \
            --gmp_gradient_checkpointing=true \
            --gmp_max_prompt_len=512 \
            --gmp_ntp_lambda=0.33 \
            --gmp_kd_lambda=0.33 \
            --gmp_onpolicy_kd_lambda=0.33 \
            --gmp_kd_only=false \
            --gmp_onpolicy_max_new_tokens=256 \
            --gmp_opkd_prev_mask_teacher=false \
            --gmp_opkd_vllm_gpu_mem=0.15 \
            --gmp_opkd_vllm_gpu_index=0 \
            --gmp_prompt_path="$OPD_PROMPT_PATH" \
            --gmp_tr_enabled=true \
            --gmp_tr_delta_init=0.05 \
            --gmp_tr_delta_min=0.001 \
            --gmp_tr_kl_threshold=${KL_THRESHOLD} \
            --gmp_tr_kl_reduce=mean \
            --gmp_saliency=${SALIENCY} \
            --gmp_mask_interval=${MASK_INTERVAL} \
            --gmp_fisher_beta=0.999 \
            --gmp_save_path="${REPO_ROOT}/models" \
            --save_model=true \
            --push_to_hub=true \
            --eval_math500=false \
            --eval_full_bench=true \
            --eval_profile=${EVAL_PROFILE} \
            --eval_zero_shot=true \
            --wandb=true \
            --wandb_project=${WANDB_PROJECT} \
            --seed=42 \
            --run_name_suffix="${SALIENCY}_${DATA_TAG}_fsdp2multi"
        echo "=== [slot $slot] EXIT: $? ==="
    ) > "$log_path" 2>&1 &
}

idx=0
total=${#CONFIGS[@]}
while (( idx < total )); do
    pids=()
    batch_end=$(( idx + 4 < total ? idx + 4 : total ))
    for ((i = idx; i < batch_end; i++)); do
        IFS=',' read -r sp lr <<< "${CONFIGS[$i]}"
        slot=$(( i - idx ))
        run_one "$sp" "$lr" "${GPU_PAIRS[$slot]}" "$i"
        pids+=($!)
    done
    echo "=== batch [$idx..$((batch_end - 1))] launched (pids: ${pids[*]}), waiting for it to finish ==="
    wait "${pids[@]}"
    idx=$batch_end
done

echo "=== all ${total} configs finished ==="
