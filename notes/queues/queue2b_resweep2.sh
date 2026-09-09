#!/bin/bash
# queue2 continuation WITH CHECKPOINTING -- the container is scheduled to go
# down ~15:00 on 2026-09-08 and every remaining job takes ~8h (6.5h train +
# ~1.5h eval), so all of them will be killed mid-training. Checkpoints every 256
# steps (newest 1 kept, ~56 GB each) turn that from a total loss into a resume.
#
# s70_B1frozen was already launched at 10:25 without checkpointing and killed at
# step <1 to relaunch it here; nothing was lost.
#
# RESUME AFTER THE CONTAINER RETURNS: see logs/RESUME_20260908.md. Mask shards
# are rank-local, so each job must resume on the SAME GPU COUNT (2) it started on.
#
# Allocation is overlap-aware (a pair is busy if ANY of its GPUs is claimed),
# because single-GPU ALPS+SFT jobs share this box.
set -u
R2=/NHNHOME/log-postech/doyoonkim/logs/resweep2_opkdfix
DATA=/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl
PROJ=reasoning_qwen3_8b_nostrip8192
SC_U=b200_scripts/gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu.sh
SC_24=b200_scripts/gmp_pgd_grow_to_target_qwen3_8b_fsdp2gpu_24.sh
cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code
say(){ echo "[$(date -Iseconds)] $*"; }
gpus_busy(){
  local p v
  for p in $(pgrep -f "main[.]py" 2>/dev/null; pgrep -f "lighteval_patched_runner[.]py" 2>/dev/null); do
    v=$(cat /proc/$p/environ 2>/dev/null | tr '\0' '\n' | grep -m1 '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    [ -n "$v" ] && echo "$v" | tr ',' '\n'
  done | sort -u | tr '\n' ' '
}
pair_busy(){ local b=" $(gpus_busy) " g; for g in $(echo "$1" | tr ',' ' '); do case "$b" in *" $g "*) return 0;; esac; done; return 1; }

# tag : kind(u|ujump|24) : kl : port : sparsity : rollout_interval
JOBS=(
  "s70_B1frozen_delta0.02:u:0.02:29814:0.7:4096"
  "s70_A3B1jump_delta0.02:ujump:0.02:29815:0.7:4096"
  "n24_klb0.01:24:0.01:29811:-:32"
  "n24_klb0.005:24:0.005:29812:-:32"
)
launch(){ # $1=pair $2=tag $3=kind $4=kl $5=port $6=sparsity $7=ro
  local J=""; [ "$3" = "ujump" ] && J="true"
  local CK="$R2/ckpt_$2"
  mkdir -p "$CK"
  if [ "$3" = "24" ]; then
    CKPT_EVERY=256 CKPT_DIR="$CK" CUDA_VISIBLE_DEVICES="$1" setsid nohup bash "$SC_24" \
      "$4" "$5" 512 32 cosine 2048 1e-4 "$DATA" 8192 true "$PROJ" \
      fisher global 0.33,0.33,0.33 "$7" 0 4 8 0.20 > "$R2/$2.log" 2>&1 &
  else
    JUMP="${J:-false}" CKPT_EVERY=256 CKPT_DIR="$CK" CUDA_VISIBLE_DEVICES="$1" setsid nohup bash "$SC_U" \
      "$6" "$4" "$5" 512 32 cosine 2048 1e-4 "$DATA" 8192 true "$PROJ" \
      fisher global 0.33,0.33,0.33 "$7" 0 4 8 0.20 > "$R2/$2.log" 2>&1 &
  fi
  say "START $2 pid=$! gpu=$1 kl=$4 ro=$7 jump=${J:-false} ckpt_every=256 ckpt_dir=$CK"
}
say "queue2b armed: ${#JOBS[@]} jobs, checkpointing ON (every 256 steps)"
i=0
while [ "$i" -lt "${#JOBS[@]}" ]; do
  for pair in "0,1" "2,3"; do
    [ "$i" -lt "${#JOBS[@]}" ] || break
    pair_busy "$pair" && continue
    IFS=':' read -r TAG KIND KL PORT SP RO <<< "${JOBS[$i]}"
    launch "$pair" "$TAG" "$KIND" "$KL" "$PORT" "$SP" "$RO"
    i=$((i+1)); sleep 180
  done
  [ "$i" -lt "${#JOBS[@]}" ] && sleep 120
done
say "queue2b: all ${#JOBS[@]} launched; runner exiting"
