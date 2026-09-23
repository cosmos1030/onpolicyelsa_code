#!/bin/bash
#SBATCH --job-name=alps_pgd_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n84,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# ALPS mask + PGD, 1.7B. Fork of slurm_alps_sft_ntpkd_opkd_qwen3_1.7b.sh
# (which retrains on a FROZEN ALPS mask) with PGD turned on, so the mask can
# move from the ALPS starting point instead of staying where ALPS put it.
# The 4B analogue is b200_scripts/alps_pgd_qwen3_4b.sh; this is the cluster
# side of the same experiment.
#
# gmp_fixed_mask=true only seeds the mask from the checkpoint's zeros and
# disables schedule-driven growth -- it does NOT gate PGD. So with
# gmp_pgd=true the arm starts at the ALPS mask and lets the self-KL-gated
# prune/revive search re-select weights from there.
#
# Two arms, set by KL_BUDGET:
#   KL_BUDGET=0.01  (default)  trust region ON -- matches 1.7B SCOUT s70
#   KL_BUDGET=99999            trust region removed; the bisection accepts
#                              every prune candidate it can within
#                              gmp_pgd_kl_bisect_iters=6 steps of doubling
#
# ARM_TAG lands in the run name so the two arms cannot collide (that bug cost
# us a checkpoint once -- see feedback_run_naming).
#
# Usage:
#   sbatch scripts/slurm_alps_pgd_qwen3_1.7b.sh 0.7 1e-4
#   KL_BUDGET=99999 ARM_TAG=pgd_notr sbatch scripts/slurm_alps_pgd_qwen3_1.7b.sh 0.7 1e-4

SPARSITY=${1:?"Usage: sbatch slurm_alps_sft_ntpkd_opkd_qwen3_1.7b.sh <SPARSITY> [LR] [SPARSITY_TYPE] [OPD_GEN_LEN] [LR_SCHEDULER] [DATA_PATH] [SEQLEN] [WANDB_PROJECT]"}
LR=${2:-1e-4}
SPARSITY_TYPE=${3:-unstructured}
OPD_GEN_LEN=${4:-512}
LR_SCHEDULER=${5:-cosine}
WANDB_PROJECT=${8:-reasoning_qwen3_1.7b_nostrip8192}
LOSS_WEIGHTS=${9:-0.33,0.33,0.33}  # NTP,KD,OPKD -- e.g. 0,0.5,0.5 to drop NTP and split KD/OPKD evenly
KL_BUDGET="${KL_BUDGET:-0.01}"   # 1.7B SCOUT s70과 동일
PGD_INTERVAL="${PGD_INTERVAL:-8}"
CALIB_SIZE="${CALIB_SIZE:-4}"
ARM_TAG="${ARM_TAG:-pgd_klb${KL_BUDGET}}"
# 984061이 in-process vLLM에서 step 46 SIGSEGV(_kl_loss 연속블록 + CuMemAllocator).
# 24cbd2c에서 검증된 처방은 sidecar + expandable_segments라 기본을 sidecar로 둔다.
# 결과에는 영향 없음(동일 설정 대조에서 시드 노이즈 안).
SIDECAR="${SIDECAR:-true}"
# Optimizer steps at which to drop an extra HF directory, e.g. "512,1024,1536".
# Empty (default) = endpoint only, i.e. the pre-existing behaviour.
MILESTONE_STEPS=${MILESTONE_STEPS:-}
# _run_tag carries only sparsity, lr and the OPKD lambda, so two arms that
# differ only in the NTP/KD split or in milestones collide in wandb. Set this.
TAG_SUFFIX=${TAG_SUFFIX:-}
# _kl_loss allocates a contiguous (1, chunk, vocab) fp32 block per chunk:
# 2048 x 151936 x 4B = 1.24GB, which PYTORCH_CUDA_ALLOC_CONF's
# max_split_size_mb:256 forbids splitting a segment to satisfy. That is the
# documented cause of the random SIGSEGV in kl_div that killed 924437
# (step 1237), 926634 (629) and 927230 (317). 256 puts the block at 156MB,
# under the limit. Chunking cuts the SEQUENCE axis while log_softmax runs
# over vocab (dim=-1), so this changes peak memory only -- the loss is the
# same number. The allocator itself is left alone because the OPD arms run
# vLLM in-process and its CuMemAllocator hard-asserts on expandable_segments.
KL_CHUNK_SIZE=${KL_CHUNK_SIZE:-256}
NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")

if [ "$SPARSITY_TYPE" = "2:4" ]; then
    ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_1.7b_alps_s24"
    SPARSITY_TAG="n24"
else
    SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")
    ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_1.7b_alps_s${SPARSITY_PCT}pct"
    SPARSITY_TAG="s${SPARSITY_PCT}pct"
fi
# --model points at the already-pruned ALPS checkpoint (the student's starting
# point); the KD/OPKD teacher must be the ORIGINAL DENSE model instead, or it
# silently becomes a frozen copy of this same pruned checkpoint (main.py's
# gmp_teacher load used to default to FLAGS.model unconditionally -- fixed to
# respect --gmp_teacher_model, see main.py `_teacher_model_path`).
DENSE_MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
DATA_PATH="${6:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}"
SEQLEN="${7:-8192}"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/wandb" "$LOCAL_JOB_BASE/slurm"
mkdir -p /home1/doyoonkim/projects/elsa/logs
NFS_LOG="/home1/doyoonkim/projects/elsa/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
# sidecar면 트레이너가 vLLM의 CuMemAllocator를 안 올리므로 expandable_segments를
# 쓸 수 있다. max_split_size_mb:256은 큰 연속 블록 분할을 금지해 이 잡엔 정반대 설정.
# ALLOC_DEFAULT=true는 B200 컨테이너와 동일한 구성이다: 이 변수를 아예 설정하지
# 않는다. B200 스크립트가 그렇게 돌고 있고 거기서는 이 segfault가 없다. 클러스터
# 쪽 max_split_size_mb:256은 원래 vLLM CuMemAllocator 우회책으로 들어온 값인데,
# 256MB 초과 세그먼트의 분할을 금지하므로 _kl_loss의 큰 연속 블록 요구와 정면
# 충돌한다(= 이 잡에 정반대 설정).
if [ "${ALLOC_DEFAULT:-false}" = "true" ]; then
    unset PYTORCH_CUDA_ALLOC_CONF
elif [ "$SIDECAR" = "true" ]; then
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
fi
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== ALPS -> Sparse SFT NTP+KD+OPKD(${NTP_LAMBDA}/${KD_LAMBDA}/${OPKD_LAMBDA}) milestones=[${MILESTONE_STEPS:-none}] Qwen3-1.7B ${SPARSITY_TAG} (${SPARSITY_TYPE}) lr=${LR} opd_gen_len=${OPD_GEN_LEN} lr_scheduler=${LR_SCHEDULER} seqlen=${SEQLEN} (OT80/FW20 nostrip8192) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$ALPS_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$ALPS_MODEL" \
    --gmp_teacher_model="$DENSE_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=${SPARSITY_TYPE} \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --gmp_tr_enabled=false \
    --gmp_pgd=true \
    --gmp_pgd_grow_to_target=true \
    --gmp_pgd_kl_budget=${KL_BUDGET} \
    --gmp_pgd_interval=${PGD_INTERVAL} \
    --gmp_pgd_kl_calib_size=${CALIB_SIZE} \
    --steps=2048 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=true \
    --gmp_kl_chunk_size=${KL_CHUNK_SIZE} \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=${KD_ONLY} \
    --gmp_ntp_lambda=${NTP_LAMBDA} \
    --gmp_kd_lambda=${KD_LAMBDA} \
    --gmp_onpolicy_kd_lambda=${OPKD_LAMBDA} \
    --gmp_onpolicy_kd_interval=${ROLLOUT_INTERVAL:-32} \
    --gmp_milestone_steps="${MILESTONE_STEPS}" \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_sidecar=${SIDECAR} \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --run_name_suffix="alpspgd_${ARM_TAG}_${SPARSITY_TAG}_lr${LR}${TAG_SUFFIX}_$([ "${NTP_LAMBDA}" = "0" ] && echo kdopdonly_)$(basename "$DATA_PATH" .jsonl)" \
    --seed=42
EXIT_CODE=$?

# Propagate main.py's exit code. Without this the script always ended 0
# and sacct reported COMPLETED even when main.py had core-dumped: job
# 924437 died of the known _kl_loss SIGSEGV at step 1237/2048, wrote no
# final checkpoint, and still showed COMPLETED -- the only way to notice
# was to read the log and compare the last "Step N/2048" against 2048.
echo "=== main.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
