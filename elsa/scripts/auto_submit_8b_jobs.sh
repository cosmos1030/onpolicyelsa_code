#!/bin/bash
# Wait for Qwen3-8B download to complete, then submit dense eval + SparseGPT s50/s60/s70
# Run: nohup bash auto_submit_8b_jobs.sh > /home1/doyoonkim/projects/elsa/logs/auto_submit_8b.log 2>&1 &

LOG=/home1/doyoonkim/projects/elsa/logs/auto_submit_8b.log
DOWNLOAD_PID=3003195

echo "[$(date)] Waiting for Qwen3-8B download (PID=$DOWNLOAD_PID)..."

# Wait for download process to finish
while kill -0 $DOWNLOAD_PID 2>/dev/null; do
    sleep 15
done

echo "[$(date)] Download process exited. Verifying model..."

MODEL=$(ls -d /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*/ 2>/dev/null | head -1)
MODEL="${MODEL%/}"

if [ -z "$MODEL" ] || [ ! -f "$MODEL/config.json" ]; then
    echo "[$(date)] ERROR: Qwen3-8B config.json not found. Aborting."
    exit 1
fi

echo "[$(date)] Model found: $MODEL"

# Submit dense eval
JID_DENSE=$(sbatch /home1/doyoonkim/projects/elsa/scripts/slurm_eval_dense_qwen3_8b.sh | awk '{print $NF}')
echo "[$(date)] Submitted dense eval -> job $JID_DENSE"

# Submit SparseGPT s50, s60, s70
JID_S50=$(sbatch /home1/doyoonkim/projects/elsa/scripts/slurm_sgpt_prune_eval_qwen3_8b.sh 0.5 128 | awk '{print $NF}')
echo "[$(date)] Submitted SparseGPT s50 -> job $JID_S50"

JID_S60=$(sbatch /home1/doyoonkim/projects/elsa/scripts/slurm_sgpt_prune_eval_qwen3_8b.sh 0.6 128 | awk '{print $NF}')
echo "[$(date)] Submitted SparseGPT s60 -> job $JID_S60"

JID_S70=$(sbatch /home1/doyoonkim/projects/elsa/scripts/slurm_sgpt_prune_eval_qwen3_8b.sh 0.7 128 | awk '{print $NF}')
echo "[$(date)] Submitted SparseGPT s70 -> job $JID_S70"

echo "[$(date)] All 4 jobs submitted: dense=$JID_DENSE s50=$JID_S50 s60=$JID_S60 s70=$JID_S70"
echo "SUBMITTED dense=$JID_DENSE s50=$JID_S50 s60=$JID_S60 s70=$JID_S70"
