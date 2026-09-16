#!/bin/bash
# Score s80 ALPS+retrain milestone checkpoints on whatever GPU is free, one per
# GPU, until none are left. Written to be left running unattended and killed on
# sight: `kill $(cat /NHNHOME/log-postech/doyoonkim/logs/launch_logs/milestone_evalq.pid)`
# stops the dispatcher, and any eval already running keeps going (kill those by
# pid from the log if you want the GPU back immediately).
#
# Deliberate choices, each one a thing that has gone wrong before:
#   * --tp_size 1. The 2026-09-14 wipeout was tp=4 against a one-GPU slot: ray
#     never formed a placement group and all five benchmarks timed out, six
#     times over. The .py default is 1 now, passed here anyway so it is visible.
#   * wandb id "-". Three milestones of one run resuming into its parent run
#     overwrite each other's run.summary, which is how the trajectory gets lost.
#     Scores are read from eval_summary_resumed.json / the per-bench JSONs.
#   * --profile long, --seed 42. Seeds 0 and 1 are somebody else's box.
#   * A checkpoint with five per-benchmark result JSONs is skipped, so a restart
#     resumes rather than redoing ~4 GPU-hours.
set -u
M=/NHNHOME/log-postech/doyoonkim/models
L=/NHNHOME/log-postech/doyoonkim/logs
OUT=$L/launch_logs
MIN_STEP=${MIN_STEP:-512}
FREE_MB=${FREE_MB:-4000}
mkdir -p "$OUT"
echo $$ > "$OUT/milestone_evalq.pid"

pending() {
  # Every unscored milestone, oldest step first. The _stepNNNNNN_ segment is
  # what makes a directory a milestone at all -- only a run given
  # MILESTONE_STEPS writes one -- so no date window is needed to tell these
  # from ordinary saves. MIN_STEP drops the two 5- and 11-step directories the
  # 2026-09-13 smoke run left behind; scoring those is ~8 GPU-hours for
  # checkpoints that mean nothing. This also picks up the six 4B ALPS S70
  # milestones stranded unscored since 2026-09-14.
  shopt -s nullglob
  for d in "$M"/gmp_*_step[0-9][0-9][0-9][0-9][0-9][0-9]_*; do
    [ -f "$d/config.json" ] || continue
    st=$(basename "$d" | sed -nE 's/.*_step([0-9]{6})_.*/\1/p')
    [ -n "$st" ] || continue
    [ "$((10#$st))" -ge "$MIN_STEP" ] || continue
    n=$(find "$d/lighteval_bench" -name 'results_*.json' 2>/dev/null | wc -l)
    [ "$n" -ge 5 ] && continue
    echo "$((10#$st)) $d"
  done | sort -n | cut -d' ' -f2-
}

claimed() {
  # GPUs a live main.py holds, read from its own environment rather than from
  # nvidia-smi. Between a training run's last step and its final eval spinning
  # vLLM back up, the card reads nearly empty for minutes while the process is
  # very much still there; dispatching into that window puts two vLLM engines
  # on one GPU and OOMs both.
  for pid in $(pgrep -f 'main.py --model' 2>/dev/null); do
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
      | sed -nE 's/^CUDA_VISIBLE_DEVICES=(.*)$/\1/p' | tr ',' '\n'
  done | sort -u
}

free_gpu() {                     # first GPU under FREE_MB and claimed by nobody
  busy=$(claimed | tr '\n' ' ')
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' -v t="$FREE_MB" -v b=" $busy " '
        $2 < t && index(b, " " $1 " ") == 0 {print $1; exit}'
}

echo "[$(date +%F' '%T)] milestone eval queue up (pid $$). pending: $(pending | wc -l)"
while true; do
  todo=$(pending | head -1)
  if [ -z "$todo" ]; then
    echo "[$(date +%F' '%T)] nothing pending; sleeping 10m (milestones are still being written)"
    sleep 600
    [ -z "$(pending)" ] && [ -z "$(pgrep -f 'main.py --model')" ] && {
      echo "[$(date +%F' '%T)] no pending checkpoints and no training left -- done"; break; }
    continue
  fi
  g=$(free_gpu)
  if [ -z "$g" ]; then sleep 300; continue; fi
  name=$(basename "$todo")
  echo "[$(date +%F' '%T)] GPU $g <- $name"
  CUDA_VISIBLE_DEVICES="$g" setsid nohup bash b200_scripts/resume_eval_lighteval.sh \
      "$todo" - --tp_size 1 --profile long --seed 42 --no_hub \
      > "$OUT/mseval_${name}.log" 2>&1 &
  echo "    pid $! -> $OUT/mseval_${name}.log"
  sleep 180                      # let it claim the GPU before the next dispatch
done
