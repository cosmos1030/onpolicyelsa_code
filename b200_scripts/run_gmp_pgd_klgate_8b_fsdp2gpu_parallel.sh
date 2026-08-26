#!/bin/bash
# Runs the 8B counterpart of run_gmp_pgd_klgate_4b_parallel.sh's wave1
# (PGD active during growth too, i.e. gmp_pgd_post_target_only=false) --
# s50/s60/s70 unstructured + s50 2:4, same lr/kl_threshold pattern as the
# 1.7B/4B families (s50: lr=5e-5 kl=0.01, others: lr=1e-4 kl=0.02), same
# kl_budget=999-uncapped-but-logged trick + pgd_interval=8 + mi=32.
#
# 8B needs FSDP 2-GPU (single-GPU 8B OOMs on the full-vocab KD loss --
# see gmp_pgd_klgate_qwen3_8b_fsdp2gpu.sh's header), so this container's 4
# GPUs only fit 2 concurrent 8B jobs (2 GPUs each), not 4 like the 4B/1.7B
# single-GPU recipe -- runs in 2 waves of 2, CUDA_VISIBLE_DEVICES-pinned to
# disjoint GPU pairs, each with its own --master_port so two independent
# torchrun rendezvous don't collide on localhost.
#
# wave2 (post_target_only=true) intentionally NOT included here -- run
# run_gmp_pgd_klgate_8b_fsdp2gpu_parallel_wave2.sh separately once this
# wave's results are reviewed (matches how 4B wave1/wave2 were done as two
# separate decisions, not one auto-chained script).
#
# **This container has no SLURM** -- run directly with `bash`. Machine-local
# launcher (paths under /NHNHOME/log-postech/doyoonkim/).
#
# Usage: nohup bash b200_scripts/run_gmp_pgd_klgate_8b_fsdp2gpu_parallel.sh > /NHNHOME/log-postech/doyoonkim/logs/queue_gmp_pgd_klgate_8b/parallel.out 2>&1 &
set -u

DATA="${OT3_DATA:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}"
PROJ=reasoning_qwen3_8b_nostrip8192
SCRIPT="$(dirname "$0")/gmp_pgd_klgate_qwen3_8b_fsdp2gpu.sh"
QUEUE_LOG_DIR="/NHNHOME/log-postech/doyoonkim/logs/queue_gmp_pgd_klgate_8b"
mkdir -p "$QUEUE_LOG_DIR"

# label|gpu_pair|master_port|args (args passed as-is to gmp_pgd_klgate_qwen3_8b_fsdp2gpu.sh, MASTER_PORT arg filled in below)
# NOTE: "BATCH1"/"BATCH2" here just means "1st/2nd pair of concurrent GPU
# slots" -- unrelated to the 4B run's wave1(duringgrowth)/wave2(post_target_only)
# terminology. All 4 jobs below are duringgrowth (PGD active during growth
# too); post_target_only is a separate script, not scheduled here.
BATCH1=(
  "s50_u_duringgrowth|0,1|29500|0.5 999 0.01 __PORT__ 512 32 cosine 2048 0 5e-5 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8"
  "s60_u_duringgrowth|2,3|29501|0.6 999 0.02 __PORT__ 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8"
)
BATCH2=(
  "s70_u_duringgrowth|0,1|29500|0.7 999 0.02 __PORT__ 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8"
  "s50_24_duringgrowth|2,3|29501|0.5 999 0.02 __PORT__ 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 2:4 0.0 32 0 4 false 8"
)

run_wave() {
  local wave_name="$1"; shift
  local jobs=("$@")
  local pids=()
  local labels=()
  echo ""
  echo "=== [$(date -Iseconds)] ${wave_name} start: ${#jobs[@]} jobs, 2 GPUs each ==="
  for entry in "${jobs[@]}"; do
    IFS='|' read -r label gpus port args <<< "$entry"
    args="${args/__PORT__/$port}"
    job_log="${QUEUE_LOG_DIR}/${label}.log"
    echo "  -> gpus=${gpus} port=${port} ${label} -> ${job_log}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${gpus} bash "$SCRIPT" $args > "$job_log" 2>&1 &
    pids+=("$!")
    labels+=("$label")
    sleep 20  # stagger so 2 concurrent FSDP rendezvous + model-loading sequences don't slam the host at once
  done

  local i=0
  for pid in "${pids[@]}"; do
    wait "$pid"
    code=$?
    echo "=== [$(date -Iseconds)] ${wave_name} ${labels[$i]} (pid $pid): exit ${code} ==="
    i=$((i + 1))
  done
  echo "=== [$(date -Iseconds)] ${wave_name} done ==="
}

echo "=== 8B FSDP2GPU parallel queue start: $(date -Iseconds) ==="
run_wave "batch1(s50+s60,duringgrowth)" "${BATCH1[@]}"
run_wave "batch2(s70+s50_24,duringgrowth)" "${BATCH2[@]}"
echo ""
echo "=== all done: $(date -Iseconds) ==="
