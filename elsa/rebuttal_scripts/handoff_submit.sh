#!/bin/bash
# Handoff 2026-11-08: 1.7B s80 fixed-schedule baselines on the PAPER commit 5068c4c
# (paper s80 SCOUT = wandb hz4kw1sr, in-process vLLM, roll512, seed42).
# 8 jobs = {cubic, linear} x reach step E in {128,256,512,1024} (of 2048), PGD off,
# mask grown every 8 steps (--gmp_schedule_update_interval=8) while mask_interval
# stays 32, so OPD rollout refresh cadence / pool size match SCOUT exactly.
#
# Usage (from this worktree's elsa/ dir):
#   PYTHON=/path/to/env/bin/python PARTITION=<A100-80GB partition> QOS=<qos> \
#     bash rebuttal_scripts/handoff_submit.sh
# Optional: MODEL (default Qwen/Qwen3-1.7B, must resolve offline), DATA_DIR
#   (default ./data; needs ot3_fineweb_40k_qwen3_nostrip_8192.jsonl and
#   ot3_fineweb_200k_qwen3_opdprompts.jsonl), SAVE_DIR, LOCAL_BASE, EXTRA_SBATCH.
set -e
cd "$(dirname "$0")/.."
export ELSA_WT=$(pwd)
: "${PYTHON:?}" "${PARTITION:?}"
[ "$(git rev-parse --abbrev-ref HEAD)" = "rebuttal/paper5068c4c" ] || echo "WARN: not on branch rebuttal/paper5068c4c"
mkdir -p logs
for SCH in cubic linear; do for E in 128 256 512 1024; do
  R=$(python3 -c "print($E/2048)")
  N=rb_paper5068c4c_1.7b_s80_${SCH}_end${E}_upd8_inproc_roll512
  SPARSITY=0.8 EXTRA_FLAGS="--gmp_pgd=false --gmp_pgd_grow_to_target=false --gmp_sparse_train_steps=0 --gmp_pruning_end_ratio=$R --gmp_growth_schedule=$SCH --gmp_schedule_update_interval=8" \
  sbatch --parsable --export=ALL --job-name=$N --partition=$PARTITION ${QOS:+--qos=$QOS} \
    --output=$ELSA_WT/logs/%x_%j.out ${EXTRA_SBATCH:-} \
    rebuttal_scripts/portable_paper_1.7b.sh false rb_paper5068c4c_${SCH}_end${E}_upd8
done; done
