#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opkd_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_ntpkd_opkd_1.7b_%j.out
exec 2>&1

# Same as slurm_gmp_tr_kd_opkd_qwen3_1.7b.sh (job 696129, TR-GMP KD+OPKD,
# fisher saliency, kl=0.01, math500=64.8% -- best OT80/FW20 1.7B result so
# far) but with NTP added back in (gmp_kd_only=false), equal 0.33/0.33/0.33
# weighting across NTP/KD/OPKD. Testing the hypothesis that dropping NTP
# (vs. the old pre-OT80/FW20 recipe, which always included it and hit
# 64.4-67.0 math500) is what's been capping this recipe's ceiling -- NTP
# grounds the model on real CoT data directly, which reverse-KL-only KD/OPKD
# doesn't replace.
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opkd_qwen3_1.7b.sh <SPARSITY> <KL_THRESHOLD> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [POST_TARGET_STEPS] [LR]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opkd_qwen3_1.7b.sh 0.5 0.01 256
#       sbatch slurm_gmp_tr_ntpkd_opkd_qwen3_1.7b.sh 0.5 0.01 256 8 cosine 2048 8
#       sbatch slurm_gmp_tr_ntpkd_opkd_qwen3_1.7b.sh 0.5 0.02 256 32 cosine 2048 0 5e-5
#
# MASK_INTERVAL default is 8 (mask updates every 8 steps) and POST_TARGET_STEPS
# default is 8 (stop 8 steps after TR-GMP's trust-region growth ACTUALLY hits
# final_sparsity, not after a precomputed step) -- STEPS=2048 stays as an
# upper-bound safety cap, but the run will normally end much earlier. This
# tests whether the long fixed-mask "sparse training" tail (old default:
# gmp_sparse_train_steps=512, i.e. ~1/4 of the whole budget spent training
# further AFTER the mask was already final) was actively hurting rather than
# helping -- 696129 (best-overall result) and every recipe since have all used
# that same 512-step tail, so this isolates whether cutting it short recovers
# some of the gap vs ALPS's untrained one-shot checkpoint (74.6 math500).

SPARSITY=${1:?"Usage: sbatch slurm_gmp_tr_ntpkd_opkd_qwen3_1.7b.sh <SPARSITY> <KL_THRESHOLD> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [POST_TARGET_STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT]"}
KL_THRESHOLD=${2:-0.01}
OPD_GEN_LEN=${3:-256}
MASK_INTERVAL=${4:-8}
LR_SCHEDULER=${5:-cosine}
STEPS=${6:-2048}
POST_TARGET_STEPS=${7:-8}
LR=${8:-1e-4}
SEQLEN=${10:-2048}
GRAD_CKPT=${11:-false}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="${9:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_train.jsonl}"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) Qwen3-1.7B s${SPARSITY_PCT} lr=${LR} kl=${KL_THRESHOLD} opd_gen_len=${OPD_GEN_LEN} mask_interval=${MASK_INTERVAL} lr_scheduler=${LR_SCHEDULER} steps=${STEPS} post_target_steps=${POST_TARGET_STEPS} (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --do_gmp=true \
    --steps=${STEPS} \
    --gmp_post_target_steps=${POST_TARGET_STEPS} \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=${GRAD_CKPT} \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b \
    --run_name_suffix="lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}_$(basename "$DATA_PATH" .jsonl)" \
    --seed=42

echo "##### END #####"
