#!/bin/bash
#SBATCH --job-name=gmp_pgd_grow_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91,n87,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/gmp_pgd_grow_1.7b_%j.out
exec 2>&1

# PGD-driven growth fork of slurm_gmp_pgd_klgate_qwen3_1.7b.sh: no separate
# TR-GMP growth mechanism at all -- --gmp_pgd_grow_to_target=true makes
# _pgd_desired target final_sparsity directly (instead of matching current
# keep-count), and revive is no longer forced equal to prune (revive
# saturates at min(k, revive_cand) while prune keeps going up to k), so the
# self-KL-gated bisection alone drives net sparsity growth toward target, at
# whatever pace --gmp_pgd_kl_budget allows. --gmp_pruning_end_ratio=0.0
# disables the cubic/cosine schedule-driven growth path that would otherwise
# run under --gmp_tr_enabled=false (see gmp_trainer.py's mask_interval
# block: pruning_end_steps=0 makes `step <= pruning_end_steps` false from
# step 1 onward, so maskmgr.update() is never called from that path).
#
# Validated on a short (100-step, seqlen=1024, no eval) 1.7B s70 smoke test
# (job 823250, wandb run accgow1u, debug_pgd_convergence project) before
# this real launcher: sparsity climbed 0 -> exactly 0.7000 by step ~91 with
# NO overshoot, self-KL bound 35/35 (100%) of growth-phase PGD calls and
# 0/57 of post-target maintenance calls (n_prune_cand/n_revive_cand
# naturally converge once target is reached, no separate at-target branch
# needed).
#
# unstructured only -- NOT yet supported for sparsity_type=2:4/N:M, which
# would additionally need the desired-mask target to respect per-group
# structural caps while ALSO being allowed to exceed revive_cand (the N:M
# PGD paths fixed this session -- _pgd_nm_pre_target/_pgd_nm_post_target --
# still only MAINTAIN sparsity, they don't drive growth toward target).
#
# Usage: sbatch slurm_gmp_pgd_grow_to_target_qwen3_1.7b.sh <SPARSITY> <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT]
# e.g.: sbatch slurm_gmp_pgd_grow_to_target_qwen3_1.7b.sh 0.7 0.02 512 32 cosine 2048 1e-4 \
#         /home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl 8192 true reasoning_qwen3_1.7b_nostrip8192

SPARSITY=${1:?"Usage: <SPARSITY> <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT]"}
KL_BUDGET=${2:?"Usage: <SPARSITY> <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT]"}
OPD_GEN_LEN=${3:-512}
MASK_INTERVAL=${4:-32}
LR_SCHEDULER=${5:-cosine}
STEPS=${6:-2048}
LR=${7:-1e-4}
DATA_PATH_ARG=${8:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${9:-8192}
GRAD_CKPT=${10:-true}
WANDB_PROJECT=${11:-reasoning_qwen3_1.7b_nostrip8192}
SALIENCY=${12:-fisher}
PRUNING_SCOPE=${13:-global}
LOSS_WEIGHTS=${14:-0.33,0.33,0.33}  # NTP,KD,OPKD
ROLLOUT_INTERVAL=${15:-${MASK_INTERVAL}}  # gmp_onpolicy_kd_interval -- defaults to mask_interval
KD_NSAMPLES=${16:-0}  # 0 = full dataset (production)
CALIB_SIZE=${17:-4}   # gmp_pgd_kl_calib_size
PGD_INTERVAL=${18:-8}  # gmp_pgd_interval -- also the effective growth cadence in this mode (no separate mask_interval-triggered growth)
VLLM_GPU_MEM=${19:-0.15}  # gmp_opkd_vllm_gpu_mem -- 0.15 sized for OPD_GEN_LEN=256, raise for 512+
NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="$DATA_PATH_ARG"
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
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== PGD-driven growth (no TR-GMP) Qwen3-1.7B s${SPARSITY_PCT} kl_budget=${KL_BUDGET} lr=${LR} pgd_interval=${PGD_INTERVAL} lr_scheduler=${LR_SCHEDULER} steps=${STEPS} saliency=${SALIENCY} (OT80/FW20) ==="
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
    --sparsity_type=unstructured \
    --do_gmp=true \
    --steps=${STEPS} \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=${SALIENCY} \
    --gmp_pruning_scope=${PRUNING_SCOPE} \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=${GRAD_CKPT} \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=${KD_ONLY} \
    --gmp_kl_chunk_size=2048 \
    --kd_nsamples=${KD_NSAMPLES} \
    --gmp_ntp_lambda=${NTP_LAMBDA} \
    --gmp_kd_lambda=${KD_LAMBDA} \
    --gmp_onpolicy_kd_lambda=${OPKD_LAMBDA} \
    --gmp_onpolicy_kd_interval=${ROLLOUT_INTERVAL} \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=${VLLM_GPU_MEM} \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=false \
    --gmp_pruning_end_ratio=0.0 \
    --gmp_pgd=true \
    --gmp_pgd_grow_to_target=true \
    --gmp_pgd_kl_budget=${KL_BUDGET} \
    --gmp_pgd_kl_calib_size=${CALIB_SIZE} \
    --gmp_pgd_interval=${PGD_INTERVAL} \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --run_name_suffix="${RUN_TAG:+${RUN_TAG}_}pgd_grow2target_klbudget${KL_BUDGET}_lr${LR}_pgdi${PGD_INTERVAL}_${PRUNING_SCOPE}scope_$(basename "$DATA_PATH" .jsonl)" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
