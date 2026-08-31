#!/bin/bash
#SBATCH --job-name=tr_ntpkd_opkd_24_4b_pgd_fsdp4v3
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/tr_ntpkd_opkd_24_4b_pgd_fsdp4v3_%j.out
exec 2>&1

# REAL (non-smoke-test) run: TR-GMP NTP+KD+OPKD(0.33/0.33/0.33), 2:4
# semi-structured sparsity, PGD reprojection every step, structured-L1
# (bottom-2-of-4 grouping regularizer), Qwen3-4B, matched to the best mi=32
# TR-GMP+OPKD S70 baseline (wbid ef5mng93: s70, lr=5e-5, mi=32, kl=0.02,
# seqlen=8192, nostrip8192 data) except sparsity_type=2:4 + PGD + lasso.
#
# 4x A100-80GB FSDP training + 1 dedicated GPU for the OPKD vLLM rollout
# engine (5 total) -- not single-GPU like the old
# slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_pgd.sh, which predates today's fix and
# was single-GPU only because FSDP N:M/PGD support didn't exist yet. Today's
# fix (see gmp_trainer.py _fsdp_nm_reconstruct/_pgd_nm_pre_target/
# _pgd_nm_post_target/_nm_fully_closed + candidate_masks' N:M branch) makes
# mask growth, PGD's N:M-aware reprojection, and TR-GMP's structural
# completeness check all correct under FSDP1's arbitrary flat-buffer
# sharding -- validated via a real 4x A100-40GB FSDP smoke test (1.7B,
# PGD+2:4+lasso, full gradual growth + post-target PGD churn): 0.000000%
# 2:4 group violations across 352,321,536 groups in the saved checkpoint.
# structured-L1 was already FSDP-safe (rides FSDP's own pre-forward
# all-gather, see _register_structured_l1_hooks) even before today's fix.
#
# v3: shares the OPKD vLLM engine on training rank 0's own GPU
# (--gmp_opkd_vllm_gpu_index=0, tp_size=1) instead of requesting a 5th
# dedicated GPU, exactly like the working 8B fsdp2gpu ALPS->SFT script
# (slurm_alps_sft_ntpkd_opkd_qwen3_8b_fsdp2gpu.sh, gmp_opkd_vllm_gpu_index=0)
# -- that script proved tp_size=1 GPU-sharing under FSDP does NOT deadlock
# (only tp_size>1 tensor-parallel vLLM across GPUs already holding a live
# FSDP process group did, per that script's repro_nested_nccl_test.py).
# vLLM's enable_sleep_mode frees most of its footprint between rollout
# bursts, so the added transient memory on GPU0 during generation should
# fit -- but v2 (5-GPU, fully isolated vLLM) ran at ~78-79/80GB on each
# training GPU with NO vLLM sharing at all, so this is genuinely tighter
# than that proven-safe config. If GPU0 OOMs specifically during an OPKD
# rollout burst, fall back to v2 (extra dedicated GPU) rather than reducing
# training GPU count further -- see v2's own OOM lesson (2 training GPUs +
# isolated vLLM GPU already OOM'd on backward() alone, unrelated to vLLM
# placement).
#
# global batch = gmp_batch_size(1) * gmp_grad_accum(2) * world_size(4) = 8
# (matches the single-GPU PGD script's batch=1*grad_accum=8*world=1=8 and
# the FSDP2 ablation's 1*4*2=8).
#
# Usage: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_pgd_fsdp4gpu.sh <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]
# e.g.: sbatch slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_pgd_fsdp4gpu.sh 0.7 5e-5 0.02 32

SPARSITY=${1:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]"}
LR=${2:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]"}
KL_THRESHOLD=${3:?"Usage: <SPARSITY> <LR> <KL_THRESHOLD> [MASK_INTERVAL] [L1_LAMBDA] [DATA_PATH]"}
MASK_INTERVAL=${4:-32}
L1_LAMBDA=${5:-0.0001}
DATA_PATH=${6:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
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

echo "=== TR-GMP NTP+KD+OPKD(0.33/0.33/0.33) 2:4+PGD+lasso Qwen3-4B s${SPARSITY_PCT} lr=${LR} kl=${KL_THRESHOLD} mi=${MASK_INTERVAL} l1=${L1_LAMBDA}, 4xA100-80GB FSDP (vLLM shares GPU0), global_batch=8 (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=4 --master_port=${MASTER_PORT} main.py \
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
    --gmp_grad_accum=2 \
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
    --gmp_pgd=true \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=reasoning_qwen3_4b_nostrip8192 \
    --run_name_suffix="24_pgd_lr${LR}_mi${MASK_INTERVAL}_kl${KL_THRESHOLD}_fsdp4" \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
