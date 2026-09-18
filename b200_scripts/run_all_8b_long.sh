#!/bin/bash
# All twelve 8B arms, one after another -- the container has a single GPU, so
# there is nothing to parallelise.
#
# Order is s70 first, then s50, then s60: s70 is where SCOUT's margin over
# ALPS+retrain is largest and the row most likely to be asked about, so it is
# the one worth having if the machine is taken back partway through.
#
#   bash run_all_8b_long.sh                 # all twelve, 3 seeds each
#   SEEDS=42 bash run_all_8b_long.sh        # one seed -- ~3x faster, use to
#                                           # fill the table first and add
#                                           # seeds 0,1 afterwards
#   bash run_all_8b_long.sh ours_s70 alps_s70   # just these
#
# Each arm writes a done-marker and is skipped on a rerun, so this can be
# killed and restarted without redoing finished work (FORCE=1 overrides).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)

ARMS=${*:-"sparsegpt_s70 alps_s70 alpsretrain_s70 ours_s70
           sparsegpt_s50 alps_s50 alpsretrain_s50 ours_s50
           sparsegpt_s60 alps_s60 alpsretrain_s60 ours_s60"}

fail=""
for a in $ARMS; do
  echo "################ $a   ($(date '+%F %T'))"
  bash "$HERE/eval_8b_long.sh" "$a" || fail="$fail $a"
done
echo
echo "################ finished $(date '+%F %T')"
[ -n "$fail" ] && echo "failed arms:$fail" && exit 1
echo "all arms done"
