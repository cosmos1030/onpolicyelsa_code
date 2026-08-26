#!/bin/bash
# Parallel (one job per GPU) queue runner for the 8 Qwen3-4B PGD-KL-gate jobs
# on this B200 container -- this container actually has 4x B200 visible
# (confirmed via nvidia-smi 2026-08-26; b200_scripts/README.md's "single-GPU"
# framing is stale for this box), so each job is pinned to its own GPU via
# CUDA_VISIBLE_DEVICES instead of the fully-sequential
# run_gmp_pgd_klgate_4b_queue.sh. Runs in two waves of 4 (GPU count), in the
# same order as the 8 sbatch jobs on the 1.7B log_cluster run: wave 1 is the
# "PGD fires during growth too" set, wave 2 is "PGD fires only after
# final_sparsity" (post_target_only=true). Jobs within a wave run truly
# concurrently; wave 2 starts only after all of wave 1 exits.
#
# CAVEAT (untested): TRITON_CACHE_DIR/TORCHINDUCTOR_CACHE_DIR/VLLM_CACHE_ROOT
# are shared across all 4 concurrent processes (set inside
# gmp_pgd_klgate_qwen3_4b.sh, not overridden here) -- compile caches are
# normally safe to share (content-hash-keyed), but this is the first time
# this repo has run concurrent GPU jobs on this host, so watch the first
# wave's logs for cache-lock/port-collision errors before trusting wave 2.
# Machine-local launcher (paths under /NHNHOME/log-postech/doyoonkim/).
#
# Usage: nohup bash b200_scripts/run_gmp_pgd_klgate_4b_parallel.sh > /NHNHOME/log-postech/doyoonkim/logs/queue_gmp_pgd_klgate_4b/parallel.out 2>&1 &
set -u

DATA="${OT3_DATA:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}"
PROJ=reasoning_qwen3_4b_nostrip8192
SCRIPT="$(dirname "$0")/gmp_pgd_klgate_qwen3_4b.sh"
QUEUE_LOG_DIR="/NHNHOME/log-postech/doyoonkim/logs/queue_gmp_pgd_klgate_4b"
mkdir -p "$QUEUE_LOG_DIR"

WAVE1=(
  "s50_u_duringgrowth|0.5 999 0.01 512 32 cosine 2048 0 5e-5 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8"
  "s60_u_duringgrowth|0.6 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8"
  "s70_u_duringgrowth|0.7 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8"
  "s50_24_duringgrowth|0.5 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 2:4 0.0 32 0 4 false 8"
)
WAVE2=(
  "s50_u_posttargetonly|0.5 999 0.01 512 32 cosine 2048 0 5e-5 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8 true"
  "s60_u_posttargetonly|0.6 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8 true"
  "s70_u_posttargetonly|0.7 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8 true"
  "s50_24_posttargetonly|0.5 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 2:4 0.0 32 0 4 false 8 true"
)

run_wave() {
  local wave_name="$1"; shift
  local jobs=("$@")
  local pids=()
  local labels=()
  echo ""
  echo "=== [$(date -Iseconds)] ${wave_name} start: ${#jobs[@]} jobs, one per GPU ==="
  local gpu=0
  for entry in "${jobs[@]}"; do
    label="${entry%%|*}"
    args="${entry#*|}"
    job_log="${QUEUE_LOG_DIR}/${label}.log"
    echo "  -> gpu=${gpu} ${label} -> ${job_log}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=${gpu} bash "$SCRIPT" $args > "$job_log" 2>&1 &
    pids+=("$!")
    labels+=("$label")
    gpu=$((gpu + 1))
    sleep 15  # small stagger so 4 model-loading/vLLM-init sequences don't slam the host at the exact same instant
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

echo "=== parallel queue start: $(date -Iseconds) ==="
run_wave "wave1(duringgrowth)" "${WAVE1[@]}"
run_wave "wave2(posttargetonly)" "${WAVE2[@]}"
echo ""
echo "=== all done: $(date -Iseconds) ==="
