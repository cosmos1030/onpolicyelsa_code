#!/bin/bash
# Token-matched 4B 2:4 ALPS+retrain baseline.
#
# "ALPS+retrain", NOT "SFT": ALPS one-shot prune, then the mask is FROZEN
# (--gmp_fixed_mask=true) and only the weights train, under a THREE-term
# objective -- ntp_lambda=0.33 (next-token prediction) + kd_lambda=0.33
# (offline KD from the dense Qwen3-4B teacher) + onpolicy_kd_lambda=0.33
# (on-policy KD on vLLM rollouts).  Calling it "SFT" hides which term a knob
# moves, which matters here because this queue moves exactly one:
# max_new_tokens 256 -> 512 doubles the tokens per rollout in the OPKD term
# ONLY.  The rollout COUNT is 16,640 either way, and NTP and offline KD are
# untouched.
#
# Why re-run at all: the one existing 4B 2:4 ALPS+retrain run (wandb w5clve87,
# Avg 49.31) used max_new_tokens=256 and is the ONLY run in the whole 4B sweep
# that did -- every unstructured sibling used 512, and SCOUT 4B 2:4 uses 512.
# Half the on-policy KD tokens per rollout is the trap the 8B launcher's own
# comment warns about, and it runs AGAINST the baseline, so 49.31 is a floor
# and SCOUT's -2.08 deficit is a lower bound.  Only the retrain stage repeats;
# the ALPS prune is reused.
#
# Two lr arms, mirroring how the unstructured baselines were credited (best lr
# of each sparsity's sweep): 5e-5 is what the tok=256 run used, 1e-4 is what
# SCOUT 2:4 uses.
#
# PROVENANCE CAVEAT: the prune this starts from is the hub artifact
# cosmos1030/qwen3-4b-alps-2to4-ot80fw20 (pushed 2026-08-02).  The tok=256 run
# pointed at a log_cluster path (/home1/...) this container does not have, and
# the three `alps_s24` wandb runs are EVAL-ONLY, so no record proves the hub
# push is byte-identical to the prune behind 49.31.  Read the tok=512/lr=5e-5
# arm as the sanity check: it should land at or above 49.31.  A wildly
# different number means the prunes differ, not that tokens did nothing.
#
# WAITS ON EXPLICIT PIDS, NEVER `pgrep -f` PATTERNS.
set -u
LOG=/NHNHOME/log-postech/doyoonkim/logs/alps_4b_24
SC=/NHNHOME/log-postech/doyoonkim/onpolicyelsa_code
MODEL=/NHNHOME/log-postech/doyoonkim/models/qwen3_4b_alps_s24_hub
say(){ echo "[$(date -Iseconds)] $*" >> "$LOG/queue_tok512.out"; }

# PID -> GPU it hands back.  Both are the single-GPU 4B compensation runs.
declare -A WATCH=( [341077]=0 [341405]=3 )
declare -a ARMS=( "5e-05" "1e-4" )
say "armed; watching ${!WATCH[*]}; arms: ${ARMS[*]}; model=$MODEL"

if [ ! -f "$MODEL/config.json" ]; then
  say "ABORT: prune not on disk at $MODEL"; exit 1
fi
say "prune present: $(du -sh "$MODEL" | cut -f1)"

launch(){ # $1=gpu $2=lr
  local gpu="$1" lr="$2" tag="n24_tok512_lr$2"
  say "START lr=$lr tok=512 on GPU $gpu"
  cd "$SC"
  ALPS_MODEL="$MODEL" SPARSITY_TAG="$tag" CUDA_VISIBLE_DEVICES="$gpu" \
    setsid nohup bash b200_scripts/alps_sft_ntpkd_opkd_qwen3_4b_24.sh "$lr" 512 \
    > "$LOG/alps_retrain_4b_n24_tok512_lr${lr}.log" 2>&1 &
  say "  pid=$! log=$LOG/alps_retrain_4b_n24_tok512_lr${lr}.log"
}

i=0
while [ "$i" -lt "${#ARMS[@]}" ]; do
  for pid in "${!WATCH[@]}"; do
    [ "$i" -lt "${#ARMS[@]}" ] || break
    kill -0 "$pid" 2>/dev/null && continue          # still training
    gpu="${WATCH[$pid]}"
    # A job can exit while its eval subprocess still holds the GPU.
    mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if [ "$mb" -ge 2000 ]; then
      say "PID $pid gone but GPU $gpu holds ${mb} MB, waiting"
      continue
    fi
    launch "$gpu" "${ARMS[$i]}"
    unset "WATCH[$pid]"
    i=$((i+1))
    sleep 120
  done
  [ "$i" -lt "${#ARMS[@]}" ] && sleep 120
done
say "both arms launched; runner exiting"
