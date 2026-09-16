#!/bin/bash
# 4B + 2-GPU FSDP. Single-GPU 4B at seqlen 8192 OOMs on an 80GB card before it
# reaches any activation: weights 8GB + grads 8GB + Adam fp32 m/v 32GB + fp32
# master 16GB is already 64GB. FSDP shards those across the two ranks.
#SBATCH --job-name=gmp_pgd_grow_4b_fsdp4_5gpu
# 4A100 too: it takes the same hpgpu QOS, its node is 8x A100-40GB, and 4B
# under FSDP4 needs about 2GB of weights per rank plus activations, so 40GB
# is ample. It matters because every A100-80GB node is currently 8/8 GPUs
# allocated -- a 4-GPU job sits at Resources no matter its priority -- while
# 4A100 has room and 7 jobs queued against A100-80GB's 572.
# hpgpu is accepted by every 80GB-or-larger partition here (A100-80GB, 4A100,
# H200, H200-ZT, H200-PCIe-ZT), so one job can wait in all of them. It needs
# to: a 5-GPU request fits fewer holes than a 4-GPU one, and A100-80GB alone
# is 310 jobs deep. The 48GB partitions are deliberately absent -- a rank that
# peaks at 55.6 GiB cannot run on a 49 GiB card no matter where vLLM sits.
#SBATCH --partition=A100-80GB,4A100,H200,H200-ZT,H200-PCIe-ZT
#SBATCH --qos=hpgpu
# 4 GPUs, with vLLM SHARING training rank 0's card rather than taking a fifth.
# main.py's default (gmp_opkd_vllm_gpu_index=-1) puts vLLM alone on index
# world_size, which needs a 5th GPU this script never requested -- that is why
# job 930597 died in 68 seconds. Requesting 5 would work but is the untested
# path; slurm_alps_sft_ntpkd_opkd_qwen3_8b_fsdp4gpu.sh has actually run this
# sidecar with gres=gpu:4, index=0 and mem=0.3 on an 8B model, so a 4B fits
# that same arrangement with room to spare, and 4-GPU allocations schedule far
# more easily than 5.
# Two was not enough: s60 and s70 both died at step 1856 with an
# identical signature (73.10 GiB allocated, a 1.16 GiB KD-loss chunk refused),
# while s50 finished. Higher target sparsity means a larger PGD revive-candidate
# pool, so the headroom shrinks exactly where it was already thinnest. Four ranks
# halve the per-rank optimiser state again; every hyperparameter is unchanged and
# grad_accum drops 4 -> 2 so the global batch stays 8.
# 5, not 4: four training ranks plus a fifth card that belongs to vLLM alone.
# Job 938293 is the measurement. It was the first 4B SCOUT+OPD run whose flags
# actually reached main.py, and it OOMed in loss.backward() at step 8/2048
# asking for 2.32 GiB. The breakdown on GPU 0 of an 80GB A100:
#   trainer rank 0        53.29 GiB   (51.53 allocated, 0.66 reserved-unallocated)
#   vLLM sidecar          24.21 GiB   (gpu_memory_utilization 0.30)
#   ranks 1-3 NCCL bufs    1.22 GiB   (416 MiB each)
#   free                   0.51 GiB
# That is not fragmentation -- only 677 MiB was reserved-but-unallocated. The
# card is simply full, and the 2.32 GiB it wanted is exactly the vocab logits
# gradient (8192 x 151936 x 2B). Shaving vLLM's share to 0.20 buys about 8 GiB
# against a trainer that already needs 55.6, which leaves no room for the
# checkpoint gather, so the honest fix is to stop sharing the card at all.
# With gpu_index=-1 main.py puts vLLM on index world_size, i.e. the fifth GPU.
#SBATCH --gres=gpu:5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
# 16h, not 3 days. The 4B w/o-OPD runs of this script took 5:15 and 5:49;
# adding OPD's rollouts should not double that. A 3-day request cannot be
# backfilled into anything smaller than a 3-day gap, which on a partition
# 572 jobs deep is the difference between starting today and not.
# Override with --time= on the sbatch line if a run really needs longer.
#SBATCH --time=16:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/gmp_pgd_grow_4b_fsdp4_5gpu_%j.out
exec 2>&1

# FSDP fork of slurm_gmp_pgd_grow_to_target_qwen3_4b.sh: 2 training ranks
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
# the baseline -- gmp_batch_size(1) * gmp_grad_accum(2) * world_size(4) = 8,
# exactly as slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_pgd_klbudget_fsdp2gpu.sh
# does. FSDP still shards optimizer state and routes PGD's top-k through
# all_reduce, so the numeric path is not bit-identical; the engine-swap
# controls run this session put that class of difference at ~2 points of
# math500 (A: 0.106 in-process vs 0.118 sidecar; C: 0.156 vs 0.176), well
# inside the ~4.4-point seed spread and far below the effect being measured
# (baseline 0.454 vs jump 0.156).
#
# Usage: sbatch slurm_gmp_pgd_grow_to_target_qwen3_4b_fsdp2gpu.sh \
#          <SPARSITY> <KL_BUDGET> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] \
#          [STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] \
#          [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [ROLLOUT_INTERVAL] \
#          [KD_NSAMPLES] [CALIB_SIZE] [PGD_INTERVAL] [VLLM_GPU_MEM] [JUMP_TO_TARGET]
# arm B: ... 0.7 0.01 512 32 cosine 2048 1e-4 <data> 8192 true \
#          reasoning_qwen3_4b_nostrip8192 fisher global 0.33,0.33,0.33 4096 0 4 8 0.15 false

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
WANDB_PROJECT=${11:-reasoning_qwen3_4b_nostrip8192}
SALIENCY=${12:-fisher}
PRUNING_SCOPE=${13:-global}
LOSS_WEIGHTS=${14:-0.33,0.33,0.33}
ROLLOUT_INTERVAL=${15:-${MASK_INTERVAL}}
KD_NSAMPLES=${16:-0}
CALIB_SIZE=${17:-4}
PGD_INTERVAL=${18:-8}
# vLLM has its own GPU here, so this fraction is of THAT card, not shared
# with training -- it can be far more generous than the single-GPU 0.15.
# 0.3, not 0.85. main.py's comment above the launch explains why: vLLM's
# allocation on its own card gets P2P-mapped into the training GPUs' address
# space via NCCL peer access, so a large fraction there costs the trainer
# memory even though the cards are physically separate. The 8B script that
# has actually run this sidecar path uses 0.3; 0.85 was never exercised
# because every prior run of THIS script had OPD off and never launched
# vLLM at all (jobs 930597 and 934382 were the first, and both died).
# 0.15, the value the two runs that actually finished this shape used. Jobs
# 733812 and 736212 are 4B + OPKD + PGD on 4x A100-80GB FSDP with a fifth card
# for vLLM ('Launching standalone vLLM server on GPU(s) 4 ... gpu_mem=0.15'),
# and both reached Step 2048/2048 in about 10 hours. 0.3 was chosen for the
# shared-card layout and has never been exercised on a dedicated card, where
# main.py warns that vLLM's address space can be P2P-mapped into the training
# ranks. The card is no longer shared; that is the one variable being changed,
# so everything else matches the run that worked.
VLLM_GPU_MEM=${19:-0.15}
# 0 = share training rank 0's GPU (no extra card). -1 would ask for a
# dedicated GPU at index world_size, which requires --gres=gpu:5.
VLLM_GPU_INDEX=${22:--1}
JUMP_TO_TARGET=${20:-false}
# world_size=4, so grad_accum is quartered to keep global batch at 8.
GRAD_ACCUM=${21:-2}

NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_PATH="$DATA_PATH_ARG"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"

# Rescue the vLLM sidecar's own log. vllm_proc.py writes it to the compute
# node's /tmp, so when the sidecar fails to come up the only record of WHY
# dies with the allocation -- jobs 930597 and 934382 both failed there and
# left nothing to read but "did not become ready within 480s". Copy whatever
# vllm_server_*.log this job produced back to NFS on exit.
_VLLM_LOG_DEST=/home1/doyoonkim/projects/elsa/logs/vllm_sidecar
mkdir -p "$_VLLM_LOG_DEST"
trap 'for _l in /tmp/vllm_server_*.log; do
        [ -f "$_l" ] || continue
        cp "$_l" "$_VLLM_LOG_DEST/${SLURM_JOB_ID}_$(basename "$_l")" 2>/dev/null || true
      done' EXIT
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

echo "=== PGD-driven growth (no TR-GMP) Qwen3-4B s${SPARSITY_PCT} kl_budget=${KL_BUDGET} lr=${LR} pgd_interval=${PGD_INTERVAL} rollout_interval=${ROLLOUT_INTERVAL} jump_to_target=${JUMP_TO_TARGET} FSDP2+dedicated-vLLM-GPU global_batch=$((1*GRAD_ACCUM*4)) steps=${STEPS} (OT80/FW20) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MASTER_PORT=$MASTER_PORT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

# 256, not 2048. _kl_loss wants a contiguous (1, chunk, vocab) fp32 block
# per chunk: 2048 x 151936 x 4B = 1.24GB. With vLLM sharing rank 0's card
# at gpu_mem=0.3 (~24GB of an 80GB A100), training had 52GB and the 1.16GB
# request had nowhere to go -- that is the OOM that killed smoke job 935938
# at step 8/24. The 8B script this sidecar arrangement was borrowed from
# never set this flag, so it ran at the smaller default and never hit it.
# Chunking cuts the sequence axis while log_softmax reduces over vocab, so
# the loss is unchanged; only the peak moves.
$TORCHRUN --nproc_per_node=4 --master_port=${MASTER_PORT} main.py \
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
    --gmp_kl_chunk_size=${KL_CHUNK_SIZE:-256} \
    --kd_nsamples=${KD_NSAMPLES} \
    --gmp_ntp_lambda=${NTP_LAMBDA} \
    --gmp_kd_lambda=${KD_LAMBDA} \
    --gmp_onpolicy_kd_lambda=${OPKD_LAMBDA} \
    --gmp_onpolicy_kd_interval=${ROLLOUT_INTERVAL} \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=${VLLM_GPU_MEM} \
    --gmp_opkd_vllm_gpu_index=${VLLM_GPU_INDEX} \
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
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --run_name_suffix="${RUN_TAG:+${RUN_TAG}_}pgd_grow2target_klbudget${KL_BUDGET}_lr${LR}_pgdi${PGD_INTERVAL}_ri${ROLLOUT_INTERVAL}$([ "$JUMP_TO_TARGET" = "true" ] && echo "_jump")_fsdp2_${PRUNING_SCOPE}scope_$(basename "$DATA_PATH" .jsonl)" \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
