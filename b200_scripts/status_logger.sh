#!/bin/bash
# Append a compact status snapshot every 10 minutes to a file on shared
# storage, so that when this container is torn down (~10:30 on 2026-09-22)
# the next one can reconstruct what was running and how far it got. Lustre
# outlives the container; the session transcript does not.
set -u
R=/NHNHOME/log-postech/doyoonkim
D=$R/logs/eval_8b_long
T=$D/status_timeline.log
while :; do
  {
    echo "### $(date -Iseconds)"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/gpu /'
    for g in 0 1; do
      arm=$(grep -h "\]\[gpu$g\] START" "$D/run_4b_curve_gpu$g.log" 2>/dev/null | tail -1 | sed 's/.*START //')
      st=$(tr '\r' '\n' < "$D/run_4b_curve_gpu$g.log" 2>/dev/null | grep -oE "Step [0-9]+/2048 \| loss=[0-9.]+ \| sparsity=[0-9.]+" | tail -1)
      bn=$(tr '\r' '\n' < "$D/run_4b_curve_gpu$g.log" 2>/dev/null | grep -oE "Starting to process [0-9]+/[0-9]+ samples[^|]*\|[a-z_0-9:]+" | tail -1 | sed 's/.*|//')
      pr=$(tr '\r' '\n' < "$D/run_4b_curve_gpu$g.log" 2>/dev/null | grep -oE "Processed prompts: *[0-9]+%" | tail -1)
      echo "gpu$g arm=${arm:-?} ${st:-} bench=${bn:-} ${pr:-}"
    done
    echo "queue4b: $(awk '{print $2}' "$D/4bqueue.txt" 2>/dev/null | tr '\n' ' ')"
    echo "queue_s80: $(tr '\n' ' ' < "$D/seedqueue.txt" 2>/dev/null)"
    echo "pushed: $(cat "$D/.pushed_gpu0" "$D/.pushed_gpu1" 2>/dev/null | sed 's|.*/||' | tr '\n' ' ')"
    echo "daemons: $(pgrep -c -f 'run_4b_curve.sh|push_ckpts.sh|skip_eval_gpu' 2>/dev/null) alive"
    echo
  } >> "$T" 2>/dev/null
  sleep 600
done
