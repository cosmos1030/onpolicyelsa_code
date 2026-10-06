#!/bin/bash
# Waits for the vLLM-based Qwen3-8B self-gen trace job (794860) to finish,
# then trims its raw trace (already 1 answer/problem, n_prompts=150) down to
# 102 rows (matching the ~102 OT samples a real 128-sample/80-20 ALPS
# calibration draw uses), builds the mixed OT/FineWeb calibration jsonl, and
# launches ALPS s50/s60/s70 self-gen-v3 pruning+eval for Qwen3-8B.
set -uo pipefail

TRACE_JOB=794860
RAW_TRACE="/home1/doyoonkim/projects/elsa/data/selfgen_trace_qwen3_8b_v3_raw.jsonl"
TRIMMED_TRACE="/home1/doyoonkim/projects/elsa/data/selfgen_trace_qwen3_8b_v3_102.jsonl"
CALIB_OUT="/home1/doyoonkim/projects/elsa/data/selfgen_ot3_fineweb_qwen3_8b_8192_v3.jsonl"
MODEL_PATH="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
LOG="/home1/doyoonkim/projects/elsa/logs/chain_selfgen_v3_8b_vllm.log"

exec > "$LOG" 2>&1
echo "$(date) -- watcher started, waiting on job ${TRACE_JOB}"

while squeue -j "$TRACE_JOB" -h 2>/dev/null | grep -q .; do
    sleep 60
done
echo "$(date) -- job ${TRACE_JOB} left the queue"

if [ ! -s "$RAW_TRACE" ]; then
    echo "$(date) -- ERROR: expected raw trace file not found/empty: $RAW_TRACE"
    exit 1
fi

n_rows=$(wc -l < "$RAW_TRACE")
echo "$(date) -- raw trace has ${n_rows} rows, trimming to 102"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

head -n 102 "$RAW_TRACE" > "$TRIMMED_TRACE"
echo "$(date) -- wrote $TRIMMED_TRACE ($(wc -l < "$TRIMMED_TRACE") rows)"

echo "$(date) -- building calibration jsonl"
cd /home1/doyoonkim/projects/elsa
python3 scripts/build_selfgen_ot3_fineweb_dataset.py \
  --trace_path "$TRIMMED_TRACE" \
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
