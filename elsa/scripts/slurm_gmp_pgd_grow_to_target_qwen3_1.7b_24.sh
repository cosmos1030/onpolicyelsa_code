#!/bin/bash
#SBATCH --job-name=gmp_pgd_grow_1.7b_24
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91,n87,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/gmp_pgd_grow_1.7b_24_%j.out
exec 2>&1

# N:M (2:4) twin of slurm_gmp_pgd_grow_to_target_qwen3_1.7b.sh. N:M support
# was added to --gmp_pgd_grow_to_target this session: the pattern is treated
# as a constraint on the FINAL target only, not on intermediate masks (a
# group is free to sit at any dead-count mid-training) -- three self-KL-
# gated phases per PGD step (overshoot-prune-only / undershoot-revive-only /
# finished-group atomic-paired-swap), combined via ONE joint bisection over
# a shared fraction alpha so the whole step's self-KL is checked against the
# TRUE pre-step state, not three independently-budgeted sub-checks. Two
# runtime invariants are checked EVERY step and raise RuntimeError on
# violation: (1) no group's |alive-prune_n| ever increases, (2) the
# whole-step D_KL(before||after) is re-measured (on the exact applied
# candidate, not a re-drawn one) and must be <= budget+1e-6.
#
# Validated (100-step, seqlen=1024, 1.7B s50 2:4, job 828652) both in-loop
# (0 invariant violations across the whole run, RuntimeError count=0) and
# by an INDEPENDENT post-hoc check directly on the saved safetensors
# (fully outside the training code, no shared functions): 196/197 prunable
# tensors at exactly 0% violation (every single 4-group exactly 2-alive);
# the one non-zero tensor was embed_tokens.weight, which is outside
# maskmgr's pruning scope entirely (never touched), not a violation.
#
# Usage: sbatch slurm_gmp_pgd_grow_to_target_qwen3_1.7b_24.sh <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT]
# e.g.: sbatch slurm_gmp_pgd_grow_to_target_qwen3_1.7b_24.sh 0.02 512 32 cosine 2048 1e-4 \
#         /home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl 8192 true reasoning_qwen3_1.7b_nostrip8192

KL_BUDGET=${1:?"Usage: <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT]"}
OPD_GEN_LEN=${2:-512}
MASK_INTERVAL=${3:-32}
LR_SCHEDULER=${4:-cosine}
STEPS=${5:-2048}
LR=${6:-1e-4}
DATA_PATH_ARG=${7:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${8:-8192}
GRAD_CKPT=${9:-true}
WANDB_PROJECT=${10:-reasoning_qwen3_1.7b_nostrip8192}
SALIENCY=${11:-fisher}
PRUNING_SCOPE=${12:-global}
LOSS_WEIGHTS=${13:-0.33,0.33,0.33}  # NTP,KD,OPKD
ROLLOUT_INTERVAL=${14:-${MASK_INTERVAL}}  # gmp_onpolicy_kd_interval -- defaults to mask_interval (ro=32 here)
KD_NSAMPLES=${15:-0}  # 0 = full dataset (production)
CALIB_SIZE=${16:-4}   # gmp_pgd_kl_calib_size
PGD_INTERVAL=${17:-8}  # gmp_pgd_interval
VLLM_GPU_MEM=${18:-0.15}  # gmp_opkd_vllm_gpu_mem
NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")

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

echo "=== PGD-driven growth (no TR-GMP) Qwen3-1.7B 2:4 kl_budget=${KL_BUDGET} lr=${LR} pgd_interval=${PGD_INTERVAL} mask_interval=${MASK_INTERVAL} rollout_interval=${ROLLOUT_INTERVAL} lr_scheduler=${LR_SCHEDULER} steps=${STEPS} saliency=${SALIENCY} (OT80/FW20) ==="
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
    --sparsity_ratio=0.5 \
    --sparsity_type=2:4 \
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
    --run_name_suffix="${RUN_TAG:+${RUN_TAG}_}pgd_grow2target_24_klbudget${KL_BUDGET}_lr${LR}_pgdi${PGD_INTERVAL}_${PRUNING_SCOPE}scope_$(basename "$DATA_PATH" .jsonl)" \
    --seed=42

EXIT_CODE=$?
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
