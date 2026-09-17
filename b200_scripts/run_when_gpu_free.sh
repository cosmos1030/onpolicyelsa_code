#!/bin/bash
# Hold a command until GPUs come free, then run it on them.
#
#   NEED=1 bash b200_scripts/run_when_gpu_free.sh bash b200_scripts/foo.sh <args>
#
# NEED (default 1) is how many GPUs the command wants; they are exported as
# CUDA_VISIBLE_DEVICES. Everything else in the environment passes straight
# through, so TAG_SUFFIX/CKPT_EVERY/... are set the same way as on a direct run.
#
# Free means: under FREE_MB *and* not claimed by a live main.py or evaluator.
# The second half matters -- between a training run's last step and its final
# eval bringing vLLM back up, and again during LCB's CPU-side scoring, a card
# reads nearly empty for minutes while the process still owns it. Taking it
# then puts two engines on one GPU.
set -u
NEED=${NEED:-1}
FREE_MB=${FREE_MB:-4000}
POLL=${POLL:-120}

claimed() {
  for pid in $(pgrep -f 'main.py --model|lighteval_patched_runner' 2>/dev/null); do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
      | sed -nE 's/^CUDA_VISIBLE_DEVICES=(.*)$/\1/p' | tr ',' '\n'
  done | sort -u
}

free_gpus() {
  busy=$(claimed | tr '\n' ' ')
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v t="$FREE_MB" -v b=" $busy " \
        '$2 < t && index(b, " " $1 " ") == 0 {print $1}'
}

echo "[$(date +%F' '%T)] waiting for $NEED free GPU(s) to run: $*"
while true; do
  g=$(free_gpus | head -"$NEED" | paste -sd, -)
  n=$(printf '%s' "$g" | awk -F, '{print NF}')
  [ -n "$g" ] && [ "$n" -eq "$NEED" ] && break
  sleep "$POLL"
done
echo "[$(date +%F' '%T)] GPU $g free -- launching"
export CUDA_VISIBLE_DEVICES="$g"
exec "$@"
