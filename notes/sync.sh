#!/bin/bash
# Re-harvest and refresh notes/ from the live logs directory.
# Deliberately does NOT commit or push: these files carry claims about
# experiments, and an unattended commit loop would publish them unreviewed.
set -eu
L=/NHNHOME/log-postech/doyoonkim/logs
R="$(cd "$(dirname "$0")/.." && pwd)"
N="$R/notes"
PY=/NHNHOME/log-postech/doyoonkim/miniconda3/envs/rac/bin/python

cd "$L"
"$PY" harvest_results.py
if [ -f /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key ]; then
  WANDB_API_KEY=$(cat /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key) \
  WANDB_SILENT=true "$PY" harvest_wandb.py || echo "  (wandb harvest skipped)"
fi

cp -f "$L"/RESULTS_SESSION_*.md "$L"/NEXT_SESSION.md "$L"/RESUME_*.md \
      "$L"/PLAN_*.md "$N"/ 2>/dev/null || true
cp -f "$L"/harvest_results.py "$L"/harvest_wandb.py "$L"/harvest_loop.sh \
      "$L"/gpu_idle_watchdog.sh "$N"/ 2>/dev/null || true
cp -f "$L"/RESULTS_ALL.tsv "$L"/RESULTS_ALL.md "$L"/RESULTS_WANDB_BASELINES.tsv \
      "$L"/hf_archive_manifest.json "$N"/ 2>/dev/null || true
cp -f "$L"/queue_*.sh "$L"/delete_verified.sh "$N"/queues/ 2>/dev/null || true
cp -f "$L"/alps_4b_24/queue_*.sh "$N"/queues/ 2>/dev/null || true

cd "$R"
echo "--- notes/ changes ---"
git status --porcelain notes/ || true
