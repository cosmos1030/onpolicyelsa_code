#!/bin/bash
#SBATCH --job-name=scout32b_4xa100
#SBATCH --partition=A100-80GB,4A100
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/scout32b_4xa100_%j.out
exec 2>&1

# Qwen3-32B SCOUT-lite (s70) on ONE node with 4x A100-80GB: 3 FSDP ranks + the vLLM rollout sidecar on GPU3.
# Same recipe as the 4xB300 cost run cost32_scout_lite (SWVr-2 table): 512 steps (HF model also saved at step
# 256), fused teacher-top-64 KD/OPD, 2-sequence KL screening, freeze after target, 2048-token rollouts,
# grad_accum 9, KL budget 0.02. Only memory placement differs -- every option below is value-preserving:
#   --gmp_offload_optimizer_fwd   Adam m/v live on the CPU (pinned)
#   --gmp_stream_optimizer_step   optimizer.step() one parameter at a time (GPU never holds all of m/v; bit-identical)
#   --gmp_fsdp_sync_every_micro   reduce-scatter grads every micro-step (no full unsharded grad on each rank)
#   --gmp_teacher_cpu_offload     teacher shards in CPU memory, copied in per layer
#   --gmp_pgd_saliency_cache_gb=0 no VRAM copy of Adam v during mask events
# On the B300 run these three alone accounted for ~43GB (no_sync grads) + ~43GB (Adam at step) + ~21GB
# (teacher shard) of the 137GB per-rank peak.
#
# Host RAM: each rank loads the 32B student and teacher on the CPU before sharding, and the pinned Adam state
# is ~43GB per rank -> use a node with >=512GB (the job takes the whole node, --mem=0).
#
# Usage:
#   sbatch a100_scripts/slurm_scout32b_4xa100.sh                 # full run (resumes from the newest checkpoint)
#   PROBE_STEPS=24 sbatch a100_scripts/slurm_scout32b_4xa100.sh  # measure memory/time over the first 24 steps
#                                                               # (3 mask events), then stop
# Extra main.py flags can be appended after the script name.

ROOT=/home1/doyoonkim/projects
CODE=${CODE:-$ROOT/elsa}
TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL=${MODEL:-Qwen/Qwen3-32B}
DATA_PATH=${DATA_PATH:-$ROOT/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
OPD_PROMPT_PATH=${OPD_PROMPT_PATH:-$ROOT/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl}
NAME=${NAME:-scout32b_lite_4xa100}
PROBE_STEPS=${PROBE_STEPS:-0}
CK=$ROOT/elsa/ckpt/$NAME
OUT=$ROOT/elsa/models/32b
LOG=$ROOT/elsa/logs/${NAME}_${SLURM_JOB_ID}.log
mkdir -p $CK $OUT $ROOT/elsa/logs

RESUME=$(ls $CK/step[0-9][0-9][0-9][0-9][0-9][0-9].pt 2>/dev/null | sort | tail -1)
[ "$PROBE_STEPS" -gt 0 ] && RESUME=""

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
export WANDB_DIR="$LOCAL_JOB_BASE/wandb" WANDB_SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 VLLM_HOST_IP=127.0.0.1
export NCCL_DEBUG=WARN TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600 OMP_NUM_THREADS=8
export GMP_MEM_DEBUG=1 GMP_PHASE_TIMING=1   # per-rank memory at the first steps + per-phase seconds/peaks (also to wandb)
export CUDA_VISIBLE_DEVICES=0,1,2,3

_VLLM_LOG_DEST=$ROOT/elsa/logs/vllm_sidecar
mkdir -p "$_VLLM_LOG_DEST"
trap 'for _l in /tmp/vllm_server_*.log; do [ -f "$_l" ] && cp "$_l" "$_VLLM_LOG_DEST/${SLURM_JOB_ID}_$(basename "$_l")"; done' EXIT

echo "=== $NAME job=$SLURM_JOB_ID node=$(hostname) probe=$PROBE_STEPS resume=[$RESUME] extra=[$*] $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
free -g | head -2

cd $CODE
$TORCHRUN --nproc_per_node=3 --master_port=$((29500 + SLURM_JOB_ID % 1000)) main.py \
    --model=$MODEL --dataset=mixed_cot --data_path=$DATA_PATH --gmp_prompt_path=$OPD_PROMPT_PATH \
    --sparsity_ratio=0.7 --sparsity_type=unstructured --do_gmp=true \
    --gmp_use_fsdp=true --gmp_teacher_fsdp=true \
    --gmp_offload_optimizer_fwd=true --gmp_stream_optimizer_step=true --gmp_fsdp_sync_every_micro=true \
    --gmp_teacher_cpu_offload=true \
    --gmp_tr_enabled=false --gmp_pruning_end_ratio=0.0 --gmp_pgd=true --gmp_pgd_grow_to_target=true \
    --gmp_pgd_kl_budget=0.02 --gmp_pgd_kl_calib_size=2 --gmp_pgd_interval=8 --gmp_pgd_screen_source=rollout \
    --gmp_pgd_freeze_after_target=true --gmp_pgd_lazy_saliency=true --gmp_pgd_saliency_cache_gb=0 --gmp_pack_masks=true \
    --gmp_warmup_ratio=0.05 --gmp_fisher_beta=0.999 --gmp_saliency=fisher --gmp_pruning_scope=global \
    --gmp_chunked_ntpkd_loss=false --gmp_fused_topk_k=64 \
    --steps=512 --gmp_batch_size=1 --gmp_grad_accum=9 --lr=1e-4 --lr_scheduler=cosine --lr_warmup_steps=64 \
    --gmp_mask_interval=32 --seqlen=8192 --gmp_gradient_checkpointing=true --gmp_kl_chunk_size=256 --gmp_max_prompt_len=512 \
    --gmp_kd_only=false --kd_nsamples=0 --gmp_ntp_lambda=0.33 --gmp_kd_lambda=0.33 --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=32 --gmp_onpolicy_max_new_tokens=2048 --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.92 --gmp_opkd_vllm_gpu_index=3 \
    --gmp_ckpt_every_steps=64 --gmp_ckpt_keep=1 --gmp_ckpt_sharded=true --gmp_ckpt_dir=$CK --gmp_resume_from="$RESUME" \
    --gmp_milestone_steps=256 --gmp_save_path=$OUT --save_model=true --push_to_hub=false \
    --eval_math500=false --eval_full_bench=false --eval_zero_shot=false \
    --wandb=true --wandb_project=reasoning_qwen3_32b --seed=42 --run_name_suffix=${NAME} \
    "$@" > $LOG 2>&1 &
PID=$!

if [ "$PROBE_STEPS" -gt 0 ]; then
    until grep -aq "Step $PROBE_STEPS/512" $LOG || ! kill -0 $PID 2>/dev/null; do sleep 30; done
    kill -0 $PID 2>/dev/null && { pkill -TERM -P $PID; kill -TERM $PID; sleep 60; pkill -9 -P $PID; pkill -9 -f vllm_server_standalone; }
    echo "=== probe stopped at $(grep -aoE 'Step [0-9]+/512' $LOG | tail -1) ==="
    grep -aE "\[mem\]|\[pgd\]\[mem\]|phase_peak|phase_timing|OutOfMemory" $LOG | cut -c1-400 | tail -40
else
    wait $PID
    echo "=== main.py EXIT: $? $(date) ==="
fi
