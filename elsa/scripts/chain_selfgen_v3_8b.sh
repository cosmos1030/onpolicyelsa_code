#!/bin/bash
# Waits for the Qwen3-8B self-gen trace job (794637) to finish, then dedupes
# it to 102 unique problems (1 answer/problem, matching the ~102 OT samples a
# real 128-sample/80-20 ALPS calibration draw uses -- same recipe as the 1.7B
# self-gen v3 pipeline), builds the mixed OT/FineWeb calibration jsonl, and
# launches ALPS s50/s60/s70 self-gen-v3 pruning+eval for Qwen3-8B.
set -uo pipefail

TRACE_JOB=794637
MODEL_HASH="b968826d9c46dd6066d109eabc6255188de91218"
DATASET_TAG="ot3_prompts_2000_qwen3.jsonl"
TRACE_DIR="/home1/doyoonkim/projects/RAC/open-r1-main/math_trace"
TRACE_FILE="${TRACE_DIR}/dataset_${MODEL_HASH}_trace_${DATASET_TAG}_.jsonl"
DEDUP_FILE="${TRACE_DIR}/dataset_${MODEL_HASH}_trace_ot3_v3_dedup102.jsonl"
CALIB_OUT="/home1/doyoonkim/projects/elsa/data/selfgen_ot3_fineweb_qwen3_8b_8192_v3.jsonl"
MODEL_PATH="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/${MODEL_HASH}"
LOG="/home1/doyoonkim/projects/elsa/logs/chain_selfgen_v3_8b.log"

exec > "$LOG" 2>&1
echo "$(date) -- watcher started, waiting on job ${TRACE_JOB}"

while squeue -j "$TRACE_JOB" -h 2>/dev/null | grep -q .; do
    sleep 60
done
echo "$(date) -- job ${TRACE_JOB} left the queue"

if [ ! -f "$TRACE_FILE" ]; then
    echo "$(date) -- ERROR: expected trace file not found: $TRACE_FILE"
    find "$TRACE_DIR" -maxdepth 1 -newer /home1/doyoonkim/projects/RAC/open-r1-main/logs_trace -iname "*${DATASET_TAG}*"
    exit 1
fi

echo "$(date) -- found trace file, dedup + sample 102"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

python3 - << PYEOF
import json, random

rows = [json.loads(l) for l in open("${TRACE_FILE}")]
print("total rows", len(rows))

seen = {}
for r in rows:
    orig_prompt = r["prompt"][:len(r["prompt"]) - len(r["completion"])]
    if orig_prompt not in seen:
        seen[orig_prompt] = r
uniq = list(seen.values())
print("unique problems", len(uniq))

n_ot = min(102, len(uniq))
random.Random(42).shuffle(uniq)
picked = uniq[:n_ot]
print("picked", len(picked))

with open("${DEDUP_FILE}", "w") as f:
    for r in picked:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote ${DEDUP_FILE}")
PYEOF

if [ $? -ne 0 ]; then
    echo "$(date) -- ERROR: dedup step failed"
    exit 1
fi

echo "$(date) -- building calibration jsonl"
cd /home1/doyoonkim/projects/elsa
python3 scripts/build_selfgen_ot3_fineweb_dataset.py \
  --trace_path "$DEDUP_FILE" \
  --out_path "$CALIB_OUT" \
  --model_path "$MODEL_PATH" \
  --seqlen 8192

if [ $? -ne 0 ] || [ ! -s "$CALIB_OUT" ]; then
    echo "$(date) -- ERROR: build_selfgen_ot3_fineweb_dataset.py failed or produced empty output"
    exit 1
fi
echo "$(date) -- calibration jsonl built: $CALIB_OUT"

echo "$(date) -- launching ALPS s50/s60/s70 for 8B self-gen v3"
cd /home1/doyoonkim/projects/ALPS
for s in 0.5 0.6 0.7; do
    sbatch slurm_alps_prune_8b_selfgen_v3.sh $s
done

echo "$(date) -- watcher done"
