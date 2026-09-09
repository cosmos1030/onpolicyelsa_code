#!/bin/bash
# ALPS one-shot -> SFT NTP+KD+OPKD recovery baselines, RE-RUN AT ROLLOUT LENGTH 512.
#
# WHY: the published ALPS+SFT baselines all ran gmp_onpolicy_max_new_tokens=256
# while every 8B PGD run they are compared against passed 512, so the baseline
# was training on half the on-policy KD tokens per rollout (rollout COUNT was
# matched -- 256 per 32-step refill window on both sides -- only the length
# differed). Both launchers defaulted to 256 and only our queues overrode it.
# The default is now 512 everywhere, so these invocations just take it.
#
# lr per sparsity matches whichever published baseline is being replaced:
#   s50 5e-5 (cluster nebgszrv, avg5 63.341), s60 5e-5 (B200 1v1lgjqt, 56.327),
#   s70 1e-4 (B200 gnpi8cxz, 46.402 -- beat the cluster's 5e-5 run at 44.776).
#
# SINGLE GPU per job: this path is --gmp_use_fsdp=false with vLLM in-process
# (183GB is over the ~141GB peak that forced FSDP on the cluster). Note this
# also means these runs never had the OPKD rank-duplication bug, unlike the
# 2-GPU cluster runs -- see logs/resweep2_opkdfix/README.txt.
#
# Deliberately NOT setting PYTORCH_CUDA_ALLOC_CONF: vLLM's CuMemAllocator
# (enable_sleep_mode=True) hard-asserts against expandable_segments on the
# in-process single-GPU path. The launcher relies on it being absent.
#
# GPU accounting is overlap-based, not string-equality on CUDA_VISIBLE_DEVICES:
# these are 1-GPU jobs sharing a box with 2-GPU pair jobs from queue2, and a
# "0,1" != "0" comparison would have declared a pair free while a single-GPU job
# sat on half of it.
set -u
R3=/NHNHOME/log-postech/doyoonkim/logs/alpssft512_8b
cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code
SC=b200_scripts/alps_sft_ntpkd_opkd_qwen3_8b.sh
say(){ echo "[$(date -Iseconds)] $*"; }

gpus_busy(){
  local p v
  for p in $(pgrep -f "main[.]py" 2>/dev/null; pgrep -f "lighteval_patched_runner[.]py" 2>/dev/null); do
    v=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -m1 '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2)
    [ -n "$v" ] && echo "$v" | tr ',' '\n'
  done | sort -u | tr '\n' ' '
}
all_free(){   # $1 = "0" or "0,1"
  local busy=" $(gpus_busy) " g
  for g in $(echo "$1" | tr ',' ' '); do
    case "$busy" in *" $g "*) return 1;; esac
  done
  return 0
}

# sparsity : lr : tag
JOBS=("0.5:5e-5:alpssft512_s50" "0.6:5e-5:alpssft512_s60" "0.7:1e-4:alpssft512_s70")
# low indices first, so two of these land on 0 and 1 and leave the 2,3 pair
# intact for queue2's 2-GPU jobs instead of fragmenting both pairs.
GPUS=(0 1 2 3)

say "queue3 armed: ${#JOBS[@]} ALPS+SFT baselines at rollout length 512, 1 GPU each"
i=0
while [ "$i" -lt "${#JOBS[@]}" ]; do
  for g in "${GPUS[@]}"; do
    [ "$i" -lt "${#JOBS[@]}" ] || break
    all_free "$g" || continue
    IFS=':' read -r SP LR TAG <<< "${JOBS[$i]}"
    CUDA_VISIBLE_DEVICES="$g" setsid nohup bash "$SC" "$SP" "$LR" \
      > "$R3/${TAG}.log" 2>&1 &
    say "START $TAG pid=$! gpu=$g sparsity=$SP lr=$LR (rollout 512 via launcher default)"
    i=$((i+1))
    sleep 240   # stagger: each of these builds vLLM in-process on its own GPU
  done
  [ "$i" -lt "${#JOBS[@]}" ] && sleep 120
done
say "queue3: all ${#JOBS[@]} launched; runner exiting"
