#!/bin/bash
# Cancel a multi-seed eval job once it finishes the seed it is on, because the
# remaining seeds have been relaunched as their own jobs.
#
#   nohup bash cancel_after_seed.sh <jobid> <seed-just-finished> &
#
# eval_full.py writes each seed under lighteval/seed<N>/, and logs that seed's
# metrics to wandb before starting the next one. So the first appearance of the
# NEXT seed's directory means the current seed is already safely recorded, and
# everything after it duplicates work another job is doing.
set -u
JID=${1:?"usage: cancel_after_seed.sh <jobid> <seed>"}
SEED=${2:?"usage: cancel_after_seed.sh <jobid> <seed>"}
LOGS=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs
say () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
say "watching $JID; will cancel once seed $SEED is done"
while true; do
    squeue -h -j "$JID" -o "%T" 2>/dev/null | grep -q RUNNING || { say "$JID is no longer running -- nothing to do"; exit 0; }
    f=$(ls -t "$LOGS"/*_"$JID".out 2>/dev/null | head -1)
    if [ -n "$f" ] && grep -qE "output-dir [^ ]*/lighteval/seed[0-9]+/" "$f"; then
        last=$(grep -oE "output-dir [^ ]*/lighteval/seed[0-9]+/" "$f" | grep -oE "seed[0-9]+" | tail -1)
        if [ "$last" != "seed$SEED" ]; then
            say "$JID moved on to $last -- seed $SEED is recorded, cancelling"
            scancel "$JID"
            exit 0
        fi
    fi
    sleep 120
done
