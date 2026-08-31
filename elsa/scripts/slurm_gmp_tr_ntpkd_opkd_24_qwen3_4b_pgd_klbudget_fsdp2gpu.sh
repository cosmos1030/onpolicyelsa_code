#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opkd_24_4b_pgd_fsdp2
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_ntpkd_opkd_24_4b_pgd_fsdp2_%j.out
exec 2>&1

# REAL (non-smoke-test) run: TR-GMP NTP+KD+OPKD(0.33/0.33/0.33), 2:4
# semi-structured sparsity, PGD reprojection every step, structured-L1
# (bottom-2-of-4 grouping regularizer), Qwen3-4B, matched to the best mi=32
# TR-GMP+OPKD S70 baseline (wbid ef5mng93: s70, lr=5e-5, mi=32, kl=0.02,
# seqlen=8192, nostrip8192 data) except sparsity_type=2:4 + PGD + lasso.
#
# 2x A100-80GB FSDP training + 1 dedicated GPU for the OPKD vLLM rollout
# engine (not single-GPU like the old slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_pgd.sh,
# which predates today's fix and was single-GPU only because FSDP N:M/PGD
# support didn't exist yet). Today's fix (see gmp_trainer.py
# _fsdp_nm_reconstruct/_pgd_nm_pre_target/_pgd_nm_post_target +
# candidate_masks' N:M branch) makes both mask growth and PGD's N:M-aware
# reprojection correct under FSDP1's arbitrary flat-buffer sharding --
# validated via a real 4x A100-40GB FSDP smoke test (1.7B, PGD+2:4+lasso,
# full gradual growth + post-target PGD churn): 0.000000% 2:4 group
# violations across 352,321,536 groups in the saved checkpoint.
# structured-L1 was already FSDP-safe (rides FSDP's own pre-forward
# all-gather, see _register_structured_l1_hooks) even before today's fix.
#
# 3 total GPUs, not 4: under FSDP, main.py launches the OPKD vLLM engine as
# a FULLY SEPARATE standalone process on its own dedicated GPU (index
# world_size, i.e. index 2 here for a 2-GPU training world), never sharing
# memory with the training ranks at all (unlike the single-GPU script,
# where vLLM shares the SAME GPU via --gmp_opkd_vllm_gpu_mem as a memory
# fraction) -- a first attempt at this script requested only 4 GPUs for a
# 4-GPU training world and crashed immediately (`vLLM server process exited
# early`) because GPU index 4 didn't exist in the allocation. Since vLLM's
# memory footprint is now fully isolated from the 2 training GPUs, the
# training-GPU memory profile is essentially identical to the NTP+KD-only
# ablation's (slurm_gmp_tr_ntpkd_only_qwen3_4b_fsdp2gpu.sh, which fit fine
# on 2x80GB) plus PGD's modest per-step Fisher-scratch buffers and
# structured-L1's grouping term -- 2 training GPUs should have plenty of
# headroom.
#
# global batch = gmp_batch_size(1) * gmp_grad_accum(4) * world_size(2) = 8
# (matches the single-GPU PGD script's batch=1*grad_accum=8*world=1=8 and
# the FSDP2 ablation's 1*4*2=8).
#
# Fork of slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_pgd_fsdp2gpu.sh: that script
# hardcoded gen_len=256 and never exposed --gmp_pgd_kl_budget at all (so PGD
# ran fully uncapped despite --gmp_pgd=true), which doesn't match the
# capped/gen_len=512 recipe used everywhere else on the dashboard. Adds
# KL_BUDGET (capped self-KL bisection) and OPD_GEN_LEN as real params;
# L1_LAMBDA now defaults to 0.0 (no lasso, matching the other capped 2:4
# entries) instead of 0.0001.
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_pgd_klbudget_fsdp2gpu.sh <SPARSITY> <LR> <KL_THRESHOLD> <KL_BUDGET> [MASK_INTERVAL] [OPD_GEN_LEN] [L1_LAMBDA] [DATA_PATH]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_pgd_klbudget_fsdp2gpu.sh 0.5 5e-5 0.02 0.02 32 512

SPARSITY=${1:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <KL_BUDGET> [MASK_INTERVAL] [OPD_GEN_LEN] [L1_LAMBDA] [DATA_PATH]"}
LR=${2:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <KL_BUDGET> [MASK_INTERVAL] [OPD_GEN_LEN] [L1_LAMBDA] [DATA_PATH]"}
KL_THRESHOLD=${3:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <KL_BUDGET> [MASK_INTERVAL] [OPD_GEN_LEN] [L1_LAMBDA] [DATA_PATH]"}
KL_BUDGET=${4:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> <KL_BUDGET> [MASK_INTERVAL] [OPD_GEN_LEN] [L1_LAMBDA] [DATA_PATH]"}
MASK_INTERVAL=${5:-32}
OPD_GEN_LEN=${6:-512}
L1_LAMBDA=${7:-0.0}
DATA_PATH=${8:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"
SEQLEN=8192

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
# expandable_segments is incompatible with vLLM's CuMemAllocator (OPKD
# sleep-mode engine hard-asserts on it) -- see infra_vllm_sleepmode_expandable_segments memory.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) 2:4+PGD+lasso Qwen3-4B s${SPARSITY_PCT} lr=${LR} kl=${KL_THRESHOLD} mi=${MASK_INTERVAL} l1=${L1_LAMBDA}, 4xA100-80GB FSDP, global_batch=8 (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=2 --master_port=${MASTER_PORT} main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=2:4 \
    --gmp_l1_lambda=${L1_LAMBDA} \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --steps=2048 \
    --gmp_post_target_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --lr=${LR} \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=true \
    --gmp_kl_chunk_size=1024 \
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
    --gmp_pgd=true \
    --gmp_pgd_kl_budget=${KL_BUDGET} \
    --gmp_pgd_kl_calib_size=4 \
    --gmp_pgd_interval=8 \
    --gmp_pgd_skip_growth_step=true \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b_nostrip8192 \
    --run_name_suffix="24_pgd_klbudget${KL_BUDGET}_genlen${OPD_GEN_LEN}_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}_fsdp2" \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
