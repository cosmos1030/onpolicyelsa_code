#!/bin/bash
# ############################################################################
# SUPERSEDED -- DO NOT RUN.  Built on a false premise: that no 4B ALPS+retrain
# baseline existed.  It does (wandb w5clve87, Avg 49.31); it was trained on the
# log_cluster side, so only wandb has it, and searching local logs was not
# evidence of absence.  Re-running the prune here would have burned ~8 GPU
# hours reproducing something that already exists.
# The live successor is queue_alpssft_4b_24_tok512.sh, which reuses the prune
# and repeats only the retrain stage at max_new_tokens=512.
# ############################################################################
exit 1
# Builds the 4B 2:4 ALPS+retrain baseline that the project has never had -- every
# ALPS baseline so far is 8B, so "does SCOUT beat ALPS+retrain at 4B 2:4?"
# is currently unanswerable.  Two stages on ONE GPU: ALPS prune, then sparse
# SFT (NTP+KD+OPKD 0.33/0.33/0.33, lr=1e-4, rollout 512) -- the exact recipe
# the 8B n24 baseline (55.64) used, so the 4B comparison is apples to apples
# with SCOUT 4B 2:4 (d=0.01: 47.23, d=0.02: 45.17).
#
# WAITS ON EXPLICIT PIDS, NEVER ON `pgrep -f` PATTERNS.  A pattern wait is what
# stranded 3 GPUs for hours twice in this project: the pattern matched a second,
# longer-running job (and once matched the waiting shell's own command line).
set -u
LOG=/NHNHOME/log-postech/doyoonkim/logs/alps_4b_24
SC=/NHNHOME/log-postech/doyoonkim/onpolicyelsa_code
say(){ echo "[$(date -Iseconds)] $*" >> "$LOG/queue.out"; }

# PID -> the GPU it will hand back, read from the live queue file.
declare -A WATCH
while read -r pid tag gpus; do
  [ -z "${pid:-}" ] && continue
  case "$gpus" in *,*) continue;; esac      # only single-GPU jobs free a usable GPU alone
  WATCH[$pid]="$gpus"
done < <(sed 's/gpu//' /NHNHOME/log-postech/doyoonkim/logs/nm_compensate/RUNNING_PIDS.txt)

say "armed; watching PIDs: ${!WATCH[*]} (single-GPU jobs only)"

FREE=""
while [ -z "$FREE" ]; do
  for pid in "${!WATCH[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      FREE="${WATCH[$pid]}"
      say "PID $pid exited -> GPU $FREE is free"
      break
    fi
  done
  [ -z "$FREE" ] && sleep 120
done

# Confirm the GPU is actually idle before claiming it: the job may have exited
# while its eval subprocess still holds memory.
for _ in $(seq 1 60); do
  mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$FREE" | tr -d ' ')
  [ "$mb" -lt 2000 ] && break
  say "GPU $FREE still holds ${mb} MB, waiting"
  sleep 60
done

cd "$SC"
say "STAGE 1/2 ALPS prune on GPU $FREE"
CUDA_VISIBLE_DEVICES="$FREE" bash b200_scripts/alps_prune_qwen3_4b_24.sh \
  > "$LOG/alps_prune_4b_24.log" 2>&1
rc=$?
say "STAGE 1 exit=$rc"
if [ "$rc" -ne 0 ] || [ ! -d /NHNHOME/log-postech/doyoonkim/models/qwen3_4b_alps_s24 ]; then
  say "ABORT: prune failed or produced no model; not starting SFT"
  exit 1
fi

say "STAGE 2/2 ALPS+retrain (lr=1e-4, opd_gen_len=512) on GPU $FREE"
CUDA_VISIBLE_DEVICES="$FREE" bash b200_scripts/alps_sft_ntpkd_opkd_qwen3_4b_24.sh 1e-4 512 \
  > "$LOG/alpssft512_4b_n24.log" 2>&1
say "STAGE 2 exit=$? -- done"
