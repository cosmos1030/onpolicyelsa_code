#!/bin/bash
# Keeps RESULTS_ALL.{tsv,md} current without anyone remembering to run the
# harvester.  Re-harvests every 10 min; harvest_results.py only reads logs, so
# this is safe to run alongside training.  Container death loses the loop, not
# the results -- everything it writes is derived from the logs on disk, so one
# re-run after a restart rebuilds the whole table.
cd /NHNHOME/log-postech/doyoonkim/logs
HB=/NHNHOME/log-postech/doyoonkim/logs/harvest_loop.log
echo "=== HARVEST LOOP BOOT $(date -Iseconds) ===" >> "$HB"
while true; do
  out=$(python harvest_results.py 2>&1 | tail -1)
  echo "[$(date -Iseconds)] $out" >> "$HB"
  sleep 600
done
