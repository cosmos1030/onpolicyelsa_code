#!/bin/bash
# Fills GPU 0-3 ONLY (user constraint from 2026-09-10 08:35) as slots free.
#
# Priority, from the post-opkdfix grid's remaining holes:
#   1. 8B S70 d=0.03  -- d=0.03 was the BEST S70 cell pre-opkdfix (49.74) and
#      post-opkdfix has 0.02=50.12 / 0.05=49.11 bracketing it, so this is the
#      one hole that can still move the S70 headline number.
#   2. 8B S60 d=0.005 -- S60's post-opkdfix best is 58.90 at d=0.01; the low
#      end is untested (at S50 the low end came in BELOW d=0.01, so this is
#      completeness, not a likely winner).
#   3. 8B 2:4 d=0.03  -- completes the 2:4 delta sweep. Lowest value: 2:4
#      trails ALPS+retrain by 3.0 and a delta sweep cannot close that.
#
# Every job is 8B FSDP and needs TWO gpus; adjacency is not required, so any
# two free of {0,1,2,3} are paired. lr follows each sparsity's siblings
# (S60 5e-5, S70 1e-4, 2:4 1e-4).
#
# WAITS ON EXPLICIT PIDS, NEVER `pgrep -f` PATTERNS -- a pattern wait stranded
# GPUs twice in this project, once by matching the waiting shell itself.
set -u
L=/NHNHOME/log-postech/doyoonkim/logs
R2=$L/resweep2_opkdfix
SC=/NHNHOME/log-postech/doyoonkim/onpolicyelsa_code
DATA=/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl
PROJ=reasoning_qwen3_8b_nostrip8192
PIDF=$L/nm_compensate/RUNNING_PIDS.txt
say(){ echo "[$(date -Iseconds)] $*" >> "$L/queue_0to3.out"; }

# occupant PID -> the GPUs it will hand back
declare -A OCC=( [432648]="3" [433826]="0" [449449]="1 2" )
# tag : kind(u|24) : sparsity : kl : lr : port
JOBS=(
  "s70_delta0.03_opkdfix:u:0.7:0.03:1e-4:29891"
  "s60_delta0.005:u:0.6:0.005:5e-5:29893"
  "n24_klb0.03:24:-:0.03:1e-4:29895"
)
say "armed on GPU 0-3; occupants: ${!OCC[*]}; ${#JOBS[@]} jobs queued"

free_gpus(){                       # echo the GPUs whose owner exited AND are idle
  local out=""
  for pid in "${!OCC[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    for g in ${OCC[$pid]}; do
      local mb
      mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" | tr -d ' ')
      [ "${mb:-99999}" -lt 2000 ] && out="$out $g"
    done
  done
  echo "$out"
}

i=0
while [ "$i" -lt "${#JOBS[@]}" ]; do
  read -r -a F <<< "$(free_gpus)"
  if [ "${#F[@]}" -lt 2 ]; then sleep 120; continue; fi
  IFS=':' read -r TAG KIND SP KL LR PORT <<< "${JOBS[$i]}"
  PAIR="${F[0]},${F[1]}"
  CK="$R2/ckpt_$TAG"; mkdir -p "$CK"
  cd "$SC"
  if [ "$KIND" = "24" ]; then
    CKPT_EVERY=256 CKPT_DIR="$CK" CUDA_VISIBLE_DEVICES="$PAIR" setsid nohup bash \
      b200_scripts/gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu_24.sh \
      "$KL" "$PORT" 512 32 cosine 2048 "$LR" "$DATA" 8192 true "$PROJ" \
      fisher global 0.33,0.33,0.33 32 0 4 8 0.20 > "$R2/$TAG.log" 2>&1 &
  else
    CKPT_EVERY=256 CKPT_DIR="$CK" CUDA_VISIBLE_DEVICES="$PAIR" setsid nohup bash \
      b200_scripts/gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu.sh \
      "$SP" "$KL" "$PORT" 512 32 cosine 2048 "$LR" "$DATA" 8192 true "$PROJ" \
      fisher global 0.33,0.33,0.33 32 0 4 8 0.20 > "$R2/$TAG.log" 2>&1 &
  fi
  NEW=$!
  say "START $TAG on GPU $PAIR pid=$NEW kl=$KL lr=$LR ckpt=$CK"
  echo "$NEW $TAG gpu${F[0]},${F[1]}" >> "$PIDF"
  # the new job now owns those GPUs; drop them from every retired occupant
  for pid in "${!OCC[@]}"; do
    kill -0 "$pid" 2>/dev/null && continue
    rest=""
    for g in ${OCC[$pid]}; do
      [ "$g" = "${F[0]}" ] || [ "$g" = "${F[1]}" ] || rest="$rest $g"
    done
    OCC[$pid]="$rest"
  done
  OCC[$NEW]="${F[0]} ${F[1]}"
  i=$((i+1))
  sleep 180
done
say "all ${#JOBS[@]} launched; runner exiting"
