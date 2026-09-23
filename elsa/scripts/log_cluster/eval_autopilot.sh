#!/bin/bash
# Submit each 8B ladder arm the moment its weights finish downloading, and
# restart the downloader if it dies.
#
#   nohup bash elsa/scripts/log_cluster/eval_autopilot.sh > ~/eval_watchdog/autopilot.log 2>&1 &
#
# Pairs with eval_watchdog.sh (which kills and requeues jobs that stop making
# progress). Between them, nothing waits on a human noticing: the two batches
# lost to a wedged huggingface_hub download on 2026-09-21/22 cost 11h x 4 GPUs
# and 5h x 8 GPUs precisely because nobody was watching.
set -u
MODELS=/home/doyoonkim/models
LAUNCH=/home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/slurm_eval_8b_ladder.sh
DL=/tmp/claude-1031/-home-doyoonkim-projects-onpolicyelsa-code/8e14d134-13b3-4518-9ff9-5f2c30497d4e/scratchpad/dl4.sh
STATE=$HOME/eval_watchdog
ARMS="ours_s50 alpsretrain_s50 alps_s50 sparsegpt_s50 ours_s60 alpsretrain_s60 alps_s60 sparsegpt_s60"
mkdir -p "$STATE"
say () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

say "autopilot up"
while true; do
    pending=0
    for arm in $ARMS; do
        [ -f "$STATE/.submitted_$arm" ] && continue
        if [ -f "$MODELS/$arm/.download_complete" ]; then
            # squeue is the real check: a job may already be queued from an
            # earlier manual submit, and a duplicate would be two wandb runs
            # of the same name, which harvest splits into two half blocks.
            if squeue -u "$USER" -h -o "%j" | grep -qx "$arm"; then
                say "$arm already queued -- marking"
                touch "$STATE/.submitted_$arm"; continue
            fi
            if sbatch -J "$arm" --partition=A100,H200 "$LAUNCH" "$arm"; then
                say "submitted $arm (download complete)"
                touch "$STATE/.submitted_$arm"
            fi
        else
            pending=$((pending+1))
        fi
    done

    if [ "$pending" -gt 0 ]; then
        # The curl streams retry per file, but an exhausted retry budget or a
        # killed parent leaves arms half-fetched forever.
        if [ "$(ps -eo args | grep -c '[c]url_dl.sh')" -eq 0 ]; then
            say "downloader is gone with $pending arm(s) unfinished -- restarting"
            nohup env HF_TOKEN="${HF_TOKEN:-}" bash "$DL" >> "$STATE/dl_restart.log" 2>&1 &
            sleep 30
        fi
    else
        say "all arms downloaded and submitted -- autopilot done"
        exit 0
    fi
    sleep 120
done
