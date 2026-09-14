#!/bin/bash
# Score all six milestones on an A100 box, one checkpoint per GPU, in parallel.
# For a 4B model tp=1 has no cross-GPU comm and vLLM saturates one A100 on its
# own, so N checkpoints on N GPUs beats one checkpoint at tp=N.
#
#   export HF_TOKEN=...
#   GPUS="0 1 2 3" bash a100_scripts/run_all_milestones_a100.sh
#
# With 4 GPUs this is two waves (4 then 2). Logs: $WORK/logs/<name>.log
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
USER_NS=${USER_NS:-cosmos1030}
GPUS=${GPUS:-"0 1 2 3"}
WORK=${WORK:-$PWD/a100_eval_work}
mkdir -p "$WORK/logs"

REPOS=(
"$USER_NS/alps4b-s70-2term-step000512"
"$USER_NS/alps4b-s70-2term-step001024"
"$USER_NS/alps4b-s70-2term-step001536"
"$USER_NS/alps4b-s70-3term-step000512"
"$USER_NS/alps4b-s70-3term-step001024"
"$USER_NS/alps4b-s70-3term-step001536"
)

read -r -a G <<< "$GPUS"
n=${#G[@]}
i=0
while [ $i -lt ${#REPOS[@]} ]; do
  pids=()
  for ((j=0; j<n && i<${#REPOS[@]}; j++, i++)); do
    repo="${REPOS[$i]}"; name=$(basename "$repo")
    echo "launch $name on GPU ${G[$j]}"
    CUDA_VISIBLE_DEVICES="${G[$j]}" WORK="$WORK" TP=1 \
      bash "$HERE/eval_milestone_a100.sh" "$repo" > "$WORK/logs/$name.log" 2>&1 &
    pids+=($!)
    sleep 15          # stagger: each one spins up its own vLLM engine
  done
  echo "waiting on wave (${#pids[@]} jobs)..."
  wait "${pids[@]}"
done

echo "=== SCORES ==="
for repo in "${REPOS[@]}"; do
  name=$(basename "$repo"); f="$WORK/$name/eval_summary_resumed.json"
  if [ -f "$f" ]; then
    python - "$name" "$f" <<'PYEOF'
import json, sys
name, path = sys.argv[1], sys.argv[2]
m = json.load(open(path))
keys = ["lighteval/math500","lighteval/gpqa_diamond","lighteval/ifeval_prompt",
        "lighteval/lcb","lighteval/gsm8k"]
vals = [m.get(k) for k in keys]
avg = m.get("lighteval/avg5")
print(f"{name:38s} avg5={avg if avg is None else round(avg,2)}  " +
      "  ".join(f"{k.split('/')[1]}={'-' if v is None else round(v*100,2)}"
               for k, v in zip(keys, vals)))
PYEOF
  else
    echo "$name  NO SUMMARY -- check $WORK/logs/$name.log"
  fi
done
