#!/bin/bash
#SBATCH --job-name=tr_ntpkd_4b
#SBATCH --partition=H200
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n87
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_ntpkd_4b_%j.out
exec 2>&1

# TR-GMP NTP+KD only (NO OPD/OPKD), Fisher saliency, Qwen3-4B, OT80/FW20 data
# (ot3_fineweb_200k_qwen3_train.jsonl -- the corrected re-tokenized corpus,
# NOT any older mis-tokenized file). No checkpoint for this exact
# combination (4B, s70, NTP+KD no OPD, OT80/FW20) survived from earlier
# sessions -- old-dataset runs (jm3e1jy9/4ncquht4) were cleaned off disk and
# never pushed to HF.
#
# Single H200 (141GB), no FSDP. First attempt (702424) tried A100-80GB on
# the theory that removing OPD (no vLLM engine) would free enough headroom --
# it OOM'd at the very first TR-GMP mask update anyway: candidate_masks()
# computes Fisher scores across several candidate sparsities at once, needing
# a ~13.5GB burst on top of steady-state training memory, which pushed
# student(4B)+teacher(4B)+Adam states over 80GB. So the H200 requirement for
# 4B TR-GMP KD+OPKD (slurm_gmp_tr_kd_opkd_qwen3_4b.sh) wasn't only about
# vLLM's memory share -- this mask-search burst needs it regardless of OPD.
#
# Usage: sbatch slurm_gmp_tr_ntpkd_qwen3_4b.sh <SPARSITY> <KL_THRESHOLD>
# e.g.: sbatch slurm_gmp_tr_ntpkd_qwen3_4b.sh 0.7 0.01

SPARSITY=${1:?"Usage: sbatch slurm_gmp_tr_ntpkd_qwen3_4b.sh <SPARSITY> <KL_THRESHOLD>"}
KL_THRESHOLD=${2:-0.01}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_train.jsonl"

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
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== TR-GMP NTP+KD (no OPD) Qwen3-4B s${SPARSITY_PCT} kl=${KL_THRESHOLD} (OT80/FW20) ==="
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
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.5 \
    --gmp_kd_lambda=0.5 \
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
    --wandb_project=reasoning_qwen3_4b \
    --seed=42

echo "##### END #####"
