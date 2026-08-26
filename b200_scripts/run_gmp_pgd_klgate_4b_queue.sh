#!/bin/bash
# Sequential queue runner for the 8 Qwen3-4B PGD-KL-gate jobs on this
# single-GPU B200 container (mirrors the 8 sbatch jobs currently running as
# Qwen3-1.7B via elsa/scripts/slurm_gmp_pgd_klgate_qwen3_1.7b.sh). This
# container has no SLURM, so all 8 run one at a time through
# b200_scripts/gmp_pgd_klgate_qwen3_4b.sh -- first the 4 "PGD fires during
# growth too" configs, then the 4 "PGD fires only after final_sparsity is
# reached" (post_target_only=true) configs, in the same order as the 1.7B
# set. A failed job does NOT abort the queue (set -u only, not -e) so one
# bad config doesn't waste the remaining GPU time; each job's exit code is
# logged and a summary prints at the end.
# Machine-local launcher (paths under /NHNHOME/log-postech/doyoonkim/). Superseded
# by run_gmp_pgd_klgate_4b_parallel.sh once this container's 4 GPUs were confirmed.
#
# Usage: nohup bash b200_scripts/run_gmp_pgd_klgate_4b_queue.sh > /NHNHOME/log-postech/doyoonkim/logs/queue_gmp_pgd_klgate_4b/queue.out 2>&1 &
set -u

DATA="${OT3_DATA:-/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}"
PROJ=reasoning_qwen3_4b_nostrip8192
SCRIPT="$(dirname "$0")/gmp_pgd_klgate_qwen3_4b.sh"
QUEUE_LOG_DIR="/NHNHOME/log-postech/doyoonkim/logs/queue_gmp_pgd_klgate_4b"
mkdir -p "$QUEUE_LOG_DIR"

# label|args (args passed as-is to gmp_pgd_klgate_qwen3_4b.sh)
JOBS=(
  "s50_u_duringgrowth|0.5 999 0.01 512 32 cosine 2048 0 5e-5 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8"
  "s60_u_duringgrowth|0.6 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8"
  "s70_u_duringgrowth|0.7 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8"
  "s50_24_duringgrowth|0.5 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 2:4 0.0 32 0 4 false 8"
  "s50_u_posttargetonly|0.5 999 0.01 512 32 cosine 2048 0 5e-5 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8 true"
  "s60_u_posttargetonly|0.6 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8 true"
  "s70_u_posttargetonly|0.7 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 unstructured 0.0 32 0 4 false 8 true"
  "s50_24_posttargetonly|0.5 999 0.02 512 32 cosine 2048 0 1e-4 $DATA 8192 true $PROJ fisher global 0.33,0.33,0.33 2:4 0.0 32 0 4 false 8 true"
)

RESULTS=()
echo "=== queue start: $(date -Iseconds), ${#JOBS[@]} jobs ==="

for entry in "${JOBS[@]}"; do
  label="${entry%%|*}"
  args="${entry#*|}"
  job_log="${QUEUE_LOG_DIR}/${label}.log"
  echo ""
  echo "=== [$(date -Iseconds)] starting ${label} -> ${job_log} ==="
  # shellcheck disable=SC2086
  bash "$SCRIPT" $args > "$job_log" 2>&1
  code=$?
  echo "=== [$(date -Iseconds)] finished ${label}: exit ${code} ==="
  RESULTS+=("${label}: exit ${code}")
done

echo ""
echo "=== queue done: $(date -Iseconds) ==="
for r in "${RESULTS[@]}"; do
  echo "  $r"
done
