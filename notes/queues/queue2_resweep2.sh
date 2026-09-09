#!/bin/bash
# Rolling queue #2. Waits for queue #1's runner to finish handing out its six
# jobs before competing for GPU pairs, then uses the same rule: a pair is busy
# while ANY main.py / lighteval process still lists it in CUDA_VISIBLE_DEVICES
# (memory alone is not enough -- a finished run sits near-zero on 3 of 4 GPUs
# for ~30 min while rank 0 runs the zero-shot suite).
set -u
R2=/NHNHOME/log-postech/doyoonkim/logs/resweep2_opkdfix
DATA=/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl
PROJ=reasoning_qwen3_8b_nostrip8192
SC_U=b200_scripts/gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu.sh      # unstructured: <SPARSITY> <KL> <PORT> ...
SC_24=b200_scripts/gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu_24.sh  # 2:4:          <KL> <PORT> ...
cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code

say(){ echo "[$(date -Iseconds)] $*"; }

# Overlap-based, not string-equality on CUDA_VISIBLE_DEVICES: queue3 runs
# 1-GPU ALPS+SFT jobs on this same box, and comparing "0,1" against "0" would
# declare a pair free while a single-GPU job sat on half of it.
gpus_busy(){
  local p v
  for p in $(pgrep -f "main[.]py" 2>/dev/null; pgrep -f "lighteval_patched_runner[.]py" 2>/dev/null); do
    v=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -m1 '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    [ -n "$v" ] && echo "$v" | tr ',' '\n'
  done | sort -u | tr '\n' ' '
}
pair_busy(){   # returns 0 (busy) if ANY gpu of the pair is claimed
  local busy=" $(gpus_busy) " g
  for g in $(echo "$1" | tr ',' ' '); do
    case "$busy" in *" $g "*) return 0;; esac
  done
  return 1
}

launch(){ # $1=pair $2=tag $3=kind(u|ujump|24) $4=kl $5=port $6=sparsity $7=rollout_interval
  local ENVJUMP=""
  [ "$3" = "ujump" ] && ENVJUMP="true"
  if [ "$3" = "24" ]; then
    CUDA_VISIBLE_DEVICES="$1" setsid nohup bash "$SC_24" \
      "$4" "$5" 512 32 cosine 2048 1e-4 "$DATA" 8192 true "$PROJ" \
      fisher global 0.33,0.33,0.33 "$7" 0 4 8 0.20 > "$R2/$2.log" 2>&1 &
  else
    JUMP="${ENVJUMP:-false}" CUDA_VISIBLE_DEVICES="$1" setsid nohup bash "$SC_U" \
      "$6" "$4" "$5" 512 32 cosine 2048 1e-4 "$DATA" 8192 true "$PROJ" \
      fisher global 0.33,0.33,0.33 "$7" 0 4 8 0.20 > "$R2/$2.log" 2>&1 &
  fi
  say "START $2 pid=$! gpu=$1 kl=$4 port=$5 ro=$7 jump=${ENVJUMP:-false}"
}

# tag : kind(u|ujump|24) : kl : port : sparsity : rollout_interval
# ro=4096 (>= steps) is the B1 arm: the OPKD pool is built once and never refreshed.
JOBS=(
  "s70_delta0.01:u:0.01:29810:0.7:32"
  "s70_A3jump_delta0.02:ujump:0.02:29813:0.7:32"
  "s70_B1frozen_delta0.02:u:0.02:29814:0.7:4096"
  "s70_A3B1jump_delta0.02:ujump:0.02:29815:0.7:4096"
  "n24_klb0.01:24:0.01:29811:-:32"
  "n24_klb0.005:24:0.005:29812:-:32"
)

say "queue2 armed (${#JOBS[@]} jobs); waiting for queue1's runner to hand out its last jobs"
while pgrep -f "queue_delta_resweep[.]sh" >/dev/null 2>&1; do sleep 120; done
say "queue1 runner done -- queue2 now competing for pairs"

i=0
while [ "$i" -lt "${#JOBS[@]}" ]; do
  for pair in "0,1" "2,3"; do
    [ "$i" -lt "${#JOBS[@]}" ] || break
    pair_busy "$pair" && continue
    IFS=':' read -r TAG KIND KL PORT SP RO <<< "${JOBS[$i]}"
    launch "$pair" "$TAG" "$KIND" "$KL" "$PORT" "$SP" "$RO"
    i=$((i+1))
    sleep 180   # stagger two torchrun+vLLM inits
  done
  [ "$i" -lt "${#JOBS[@]}" ] && sleep 120
done
say "queue2: all ${#JOBS[@]} jobs launched; runner exiting"
