#!/bin/bash
# Relaunch the third ALPS+SFT-512 baseline (s70) on a single GPU.
#
# Originally started on GPU 2 at 03:43 and killed at step 26: queue3 fills GPUs
# 0,1,2,3 in order, so it took GPU 2 out of a FULLY FREE pair, leaving queue2's
# 2-GPU jobs unable to start with only GPU 3 left. 26 steps was cheap insurance
# against a GPU idling ~6h.
#
# ALLOCATION RULE: take a free GPU only when its pair-partner is already busy.
# A GPU stranded that way is worthless to queue2 (which always needs both), so
# filling it costs nothing; conversely a fully free pair is left alone so queue2
# can use it. The first version waited for a whole free pair and then took half
# of it, which was backwards twice over -- it would have re-broken good pairs,
# and it would have sat idle forever whenever only a lone GPU was free, which is
# exactly the case this job exists to fill.
#
# If no queue2 runner is alive there is nothing left that needs pairs, so any
# free GPU is fair game.
set -u
R4=/NHNHOME/log-postech/doyoonkim/logs/alpssft512_8b
cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code
say(){ echo "[$(date -Iseconds)] $*"; }
partner(){ case "$1" in 0) echo 1;; 1) echo 0;; 2) echo 3;; 3) echo 2;; esac; }
busy_gpus(){
  local p v
  for p in $(pgrep -f "main[.]py" 2>/dev/null; pgrep -f "lighteval_patched_runner[.]py" 2>/dev/null); do
    v=$(cat /proc/$p/environ 2>/dev/null | tr '\0' '\n' | grep -m1 '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    [ -n "$v" ] && echo "$v" | tr ',' '\n'
  done | sort -u | tr '\n' ' '
}
is_busy(){ case " $(busy_gpus) " in *" $1 "*) return 0;; esac; return 1; }

say "queue4 armed: will take a free GPU whose pair-partner is busy (or any free GPU once queue2 is done)"
while :; do
  q2=$(pgrep -f "queue2[.]sh" 2>/dev/null | wc -l)
  for g in 3 1 2 0; do
    is_busy "$g" && continue
    if [ "$q2" -eq 0 ] || is_busy "$(partner $g)"; then
      say "GPU $g free (partner $(partner $g) busy=$(is_busy "$(partner $g)" && echo yes || echo no), queue2 runners=$q2) -> taking it"
      CUDA_VISIBLE_DEVICES="$g" setsid nohup bash b200_scripts/alps_sft_ntpkd_opkd_qwen3_8b.sh 0.7 1e-4 \
        > "$R4/alpssft512_s70.log" 2>&1 &
      say "START alpssft512_s70 pid=$! gpu=$g sparsity=0.7 lr=1e-4 (rollout 512 via launcher default)"
      exit 0
    fi
  done
  sleep 120
done
