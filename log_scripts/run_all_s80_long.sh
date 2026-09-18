#!/bin/bash
# All three s80 arms. Sequential by default; give one GPU id per arm to run
# them side by side.
#
#   bash run_all_s80_long.sh              # one after another on the default GPU
#   bash run_all_s80_long.sh 0 1 2        # three GPUs, three arms in parallel
#
# The base arm is not optional: without it in the SAME protocol there is
# nothing to compare the DPO checkpoints against.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ARMS="base dpo_lr1e5_ep04 dpo_lr5e6_ep10"

if [ $# -eq 0 ]; then
  for a in $ARMS; do
    echo "########## $a"
    bash "$HERE/eval_s80_long.sh" "$a" || echo "!! $a failed, continuing"
  done
  exit 0
fi

i=0
set -- "$@"
pids=""
for a in $ARMS; do
  i=$((i+1))
  eval "g=\${$i:-}"
  [ -n "$g" ] || { echo "!! only $(($i-1)) GPU(s) given for 3 arms; run the rest afterwards"; break; }
  echo "########## $a on GPU $g"
  # nohup so a dropped ssh session does not take the run with it.
  nohup bash "$HERE/eval_s80_long.sh" "$a" "$g" > /dev/null 2>&1 &
  pids="$pids $!"
done
echo "launched:$pids"
echo "logs are under \$OUT_ROOT (default \$HOME/elsa_eval_s80)"
wait $pids
