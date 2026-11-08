#!/bin/bash
# Handoff 2026-11-08: 1.7B s70 SCOUT calibration-batch-size ablation on the PAPER
# commit 5a6cb61 (= code of paper s70 SCOUT wandb q9f33zl7: wandb records 92eeb8a+dirty,
# the dirty part is exactly 5a6cb61; in-process vLLM, roll512, seed42).
# 3 jobs = --gmp_pgd_kl_calib_size in {2, 8, 16} (paper = 4). Everything else = paper SCOUT.
#
# Usage (from this worktree's elsa/ dir):
#   PYTHON=/path/to/env/bin/python PARTITION=<A100-80GB partition> QOS=<qos> \
#     bash rebuttal_scripts/handoff_submit.sh
# Optional: MODEL, DATA_DIR, SAVE_DIR, LOCAL_BASE, EXTRA_SBATCH (see 5068c4c branch notes).
set -e
cd "$(dirname "$0")/.."
export ELSA_WT=$(pwd)
: "${PYTHON:?}" "${PARTITION:?}"
[ "$(git rev-parse --abbrev-ref HEAD)" = "rebuttal/paper5a6cb61" ] || echo "WARN: not on branch rebuttal/paper5a6cb61"
mkdir -p logs
for C in 2 8 16; do
  N=rb_paper5a6cb61_1.7b_s70_calib${C}_inproc_roll512
  EXTRA_FLAGS="--gmp_pgd_kl_calib_size=$C" \
  sbatch --parsable --export=ALL --job-name=$N --partition=$PARTITION ${QOS:+--qos=$QOS} \
    --output=$ELSA_WT/logs/%x_%j.out ${EXTRA_SBATCH:-} \
    rebuttal_scripts/portable_paper_1.7b_s70.sh false rb_paper5a6cb61_calib$C
done
