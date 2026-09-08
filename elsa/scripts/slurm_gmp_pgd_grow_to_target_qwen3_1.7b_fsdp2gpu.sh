#!/bin/bash
#SBATCH --job-name=gmp_pgd_grow_1.7b_fsdp2
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/gmp_pgd_grow_1.7b_fsdp2_%j.out
exec 2>&1

# FSDP fork of slurm_gmp_pgd_grow_to_target_qwen3_1.7b.sh: 2 training ranks
# plus vLLM alone on a THIRD, dedicated GPU (main.py's _use_fsdp_opkd path
# launches it on CUDA index = world_size = 2).
#
# Why: the single-GPU arm-B configuration (ROLLOUT_INTERVAL=4096,
# JUMP_TO_TARGET=false) has failed 3/3 with a SIGSEGV inside loss.backward()
# at unpredictable steps (517, 885, 929). Arms A and C, which reach target
# sparsity at step 8 via jump, complete fine -- B is the one that spends
# hundreds of steps growing with OPKD live, so it has by far the longest
# exposure.
#
# What is and is not established about that crash:
#   - "vLLM in the SAME PROCESS" was my hypothesis and is DISPROVEN: job
#     870477 used the out-of-process sidecar and still died at step 885.
#   - "vLLM on the SAME GPU" is also not sufficient to explain it: job 870446
#     shared a GPU with its sidecar and completed all 2048 steps.
#   - "vLLM on a DEDICATED GPU" is untested for this crash. The supporting
#     evidence is indirect but real: across the FSDP+OPKD runs in elsa/logs,
#     six reached 2048/2048, and NONE contains "Segmentation fault" -- the
#     ones that died did so with ordinary Python exceptions instead.
# So this is a plausible workaround, not a known fix. If it also dies in
# backward, the dedicated-GPU explanation is out too.
#
# Comparability: global batch is held at 8, matching the single-GPU arms and
# the baseline -- gmp_batch_size(1) * gmp_grad_accum(4) * world_size(2) = 8,
# exactly as slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_pgd_klbudget_fsdp2gpu.sh
# does. FSDP still shards optimizer state and routes PGD's top-k through
# all_reduce, so the numeric path is not bit-identical; the engine-swap
# controls run this session put that class of difference at ~2 points of
# math500 (A: 0.106 in-process vs 0.118 sidecar; C: 0.156 vs 0.176), well
# inside the ~4.4-point seed spread and far below the effect being measured
# (baseline 0.454 vs jump 0.156).
#
# Usage: sbatch slurm_gmp_pgd_grow_to_target_qwen3_1.7b_fsdp2gpu.sh \
#          <SPARSITY> <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] \
#          [STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] \
#          [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [ROLLOUT_INTERVAL] \
#          [KD_NSAMPLES] [CALIB_SIZE] [PGD_INTERVAL] [VLLM_GPU_MEM] [JUMP_TO_TARGET]
# arm B: ... 0.7 0.01 512 32 cosine 2048 1e-4 <data> 8192 true \
#          reasoning_qwen3_1.7b_nostrip8192 fisher global 0.33,0.33,0.33 4096 0 4 8 0.15 false

SPARSITY=${1:?"Usage: <SPARSITY> <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] ..."}
KL_BUDGET=${2:?"Usage: <SPARSITY> <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] ..."}
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
LOSS_WEIGHTS=${14:-0.33,0.33,0.33}
ROLLOUT_INTERVAL=${15:-${MASK_INTERVAL}}
KD_NSAMPLES=${16:-0}
CALIB_SIZE=${17:-4}
PGD_INTERVAL=${18:-8}
# vLLM has its own GPU here, so this fraction is of THAT card, not shared
# with training -- it can be far more generous than the single-GPU 0.15.
VLLM_GPU_MEM=${19:-0.85}
JUMP_TO_TARGET=${20:-false}
# world_size=2, so grad_accum is halved to keep global batch at 8.
GRAD_ACCUM=${21:-4}

NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
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
# This layout ALWAYS runs vLLM out of process on its own GPU, so the trainer
# never loads cumem_allocator and the assertion that forced max_split_size_mb
# on the in-process path cannot fire here. expandable_segments is the option
# that actually addresses the fragmentation that kills these runs: _kl_loss
# needs a contiguous (1, kl_chunk_size, ~152k-vocab) fp32 block (1.24GiB at
# chunk_size=2048), and max_split_size_mb:256 forbids splitting large blocks,
# which makes that harder to satisfy rather than easier.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_DEBUG=WARN
# Job 876298 died here: FSDP wrapped fine on both ranks and the dedicated-GPU
# vLLM came up, then rank 1 sat in a collective while rank 0 worked through
# the calibration/rollout setup, and NCCL's watchdog aborted at its 480s
# default ("watchdog got stuck for 480 seconds without making progress").
# This is the wait being legitimately long, not a deadlock -- the process
# group's own timeout is already 2h (main.py's init_process_group), but the
# heartbeat monitor is a SEPARATE, much shorter budget. Raise it to match.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== PGD-driven growth (no TR-GMP) Qwen3-1.7B s${SPARSITY_PCT} kl_budget=${KL_BUDGET} lr=${LR} pgd_interval=${PGD_INTERVAL} rollout_interval=${ROLLOUT_INTERVAL} jump_to_target=${JUMP_TO_TARGET} FSDP2+dedicated-vLLM-GPU global_batch=$((1*GRAD_ACCUM*2)) steps=${STEPS} (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MASTER_PORT=$MASTER_PORT"
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
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --steps=${STEPS} \
    --gmp_batch_size=1 \
    --gmp_grad_accum=${GRAD_ACCUM} \
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
    --gmp_pgd_jump_to_target=${JUMP_TO_TARGET} \
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
    --run_name_suffix="${RUN_TAG:+${RUN_TAG}_}pgd_grow2target_klbudget${KL_BUDGET}_lr${LR}_pgdi${PGD_INTERVAL}_ri${ROLLOUT_INTERVAL}$([ "$JUMP_TO_TARGET" = "true" ] && echo "_jump")_fsdp2_${PRUNING_SCOPE}scope_$(basename "$DATA_PATH" .jsonl)" \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
