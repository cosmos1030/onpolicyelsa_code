#!/bin/bash
# Alarm on IDLE GPUs, not on failed jobs. The previous watchdog only tracked job
# health, so when every job finished cleanly and a queue was blocked on a wrong
# pgrep pattern, GPUs 1-3 sat empty for hours with nothing complaining.
# Emits an ALERT line the moment >=1 of GPU 0-3 has been idle for 2 consecutive
# checks (i.e. ~4 min), and keeps emitting every 10 min while it stays idle.
set -u
HB=/NHNHOME/log-postech/doyoonkim/logs/gpu_idle_watchdog.log
INTERVAL=120
IDLE_MB=2000
declare -A strike
echo "=== GPU IDLE WATCHDOG BOOT $(date -Iseconds) host=$(hostname) watching GPU 0-3 ===" >> "$HB"
last_alert=0
while true; do
  idle=()
  for g in 0 1 2 3; do
    mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
    [ -z "$mb" ] && continue
    if [ "$mb" -lt "$IDLE_MB" ]; then
      strike[$g]=$(( ${strike[$g]:-0} + 1 ))
      [ "${strike[$g]}" -ge 2 ] && idle+=("$g")
    else
      strike[$g]=0
    fi
  done
  ts=$(date -Iseconds); now=$(date +%s)
  if [ ${#idle[@]} -gt 0 ]; then
    if [ $(( now - last_alert )) -ge 600 ]; then
      echo "ALERT $ts GPU ${idle[*]} idle (<${IDLE_MB}MB) for 2+ checks -- nothing is using $(echo ${#idle[@]}) of 4 GPUs" >> "$HB"
      last_alert=$now
    fi
  else
    echo "TICK $ts all of GPU 0-3 busy" >> "$HB"
  fi
  sleep $INTERVAL
done
