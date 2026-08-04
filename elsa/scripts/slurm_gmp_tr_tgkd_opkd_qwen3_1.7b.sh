#!/bin/bash
#SBATCH --job-name=tr_tgkd_opkd_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_tgkd_opkd_1.7b_%j.out
exec 2>&1

# TR-GMP teacher-gen-KD (forward KL, pre-generated cache) + OPD (reverse KL,
# live student rollouts) only -- no NTP, no dataset-based offline KD. Both
# loss terms share gmp_onpolicy_kd_lambda=0.5 as their weight (see
# gmp_trainer.py: TGKD's forward-KL term and OPD's reverse-KL term are both
# scaled by onpolicy_lambda -- this used to be mutually-exclusive-by-flag,
# fixed this session to allow both simultaneously).
#
# --data_path points at the TRAIN split (first 180k of the 200k-line OT80/
# FW20 corpus) -- must match scripts/slurm_pregenerate_tgkd_cache_1.7b.sh's
# --data_path/--steps/--batch_size/--grad_accum exactly, since the TGKD cache
# lookup key is data_path+gbs+n_pairs+max_new_tokens+temperature. --gmp_prompt_path
# points at the disjoint OPD-only split (last 20k lines) so OPD's rollout
# prompts and TGKD's pregenerated prompts never overlap.
#
# Run scripts/slurm_pregenerate_tgkd_cache_1.7b.sh (or wait for it to finish)
# before submitting this, or gmp_teacher_gen_kd will generate the whole cache
# inline at startup instead of loading it.
#
# Usage: sbatch slurm_gmp_tr_tgkd_opkd_qwen3_1.7b.sh <SPARSITY> <KL_THRESHOLD>
# e.g.: sbatch slurm_gmp_tr_tgkd_opkd_qwen3_1.7b.sh 0.5 0.01

SPARSITY=${1:?"Usage: sbatch slurm_gmp_tr_tgkd_opkd_qwen3_1.7b.sh <SPARSITY> <KL_THRESHOLD>"}
KL_THRESHOLD=${2:-0.01}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_train.jsonl"
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

echo "=== TR-GMP TGKD(fwdKL)+OPKD(revKL) 0.5/0.5, no NTP, Qwen3-1.7B s${SPARSITY_PCT} kl=${KL_THRESHOLD} (OT80/FW20) ==="
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
    --steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=1e-4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=32 \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --seqlen=2048 \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=true \
    --gmp_teacher_gen_kd=true \
    --gmp_dpo_cache_dir=/home1/doyoonkim/projects/elsa/.cache/dpo_chosen \
    --gmp_dpo_max_new_tokens=512 \
    --gmp_dpo_temperature=0.7 \
    --gmp_onpolicy_kd_lambda=0.5 \
    --gmp_onpolicy_max_new_tokens=256 \
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
    --seed=42

echo "##### END #####"
