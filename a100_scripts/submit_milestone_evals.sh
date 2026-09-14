#!/bin/bash
# Find milestone checkpoints written by a fixed-mask run and submit one eval job
# each, skipping any that are already scored or already queued.
#
# The save directory name is not predictable ahead of time -- _run_tag plus a
# timestamp plus the pid -- so hand-assembling these paths after a 12h run is
# both tedious and a good way to score the wrong checkpoint. This globs for
# them instead.
#
#   bash a100_scripts/submit_milestone_evals.sh [MODELS_DIR] [TAG_GLOB]
#
# TAG_GLOB narrows to one experiment, e.g. 'gmp_s70pct_lr0.0001*' for the 1.7B
# ALPS-s70 pair. Default matches every step-keyed save.
set -u
M=${1:-/home1/doyoonkim/projects/elsa/models}
GLOB=${2:-'*_step[0-9]*'}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

shopt -s nullglob
found=0
for d in "$M"/$GLOB; do
    [ -f "$d/config.json" ] || continue
    found=$((found+1))
    name=$(basename "$d")
    if [ -f "$d/eval_summary_resumed.json" ]; then
        echo "건너뜀 (이미 채점됨): $name"
        continue
    fi
    # A short job name -- SLURM truncates, and the run tag is ~50 chars.
    step=$(echo "$name" | grep -oE 'step[0-9]{6}')
    arm=$(echo "$name" | grep -q onpol_lmda && echo 3term || echo 2term)
    jn="ms17_${arm}_${step#step}"
    if squeue -u "$USER" -h -o "%j" | grep -qx "$jn"; then
        echo "건너뜀 (이미 큐에 있음): $jn"
        continue
    fi
    jid=$(sbatch --parsable --job-name="$jn" "$HERE/slurm_eval_milestone.sh" "$d")
    echo "제출 $jid  $jn  <- $name"
done
[ $found -eq 0 ] && echo "마일스톤 디렉터리를 못 찾았습니다: $M/$GLOB"
exit 0
