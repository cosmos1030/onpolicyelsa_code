#!/bin/bash
#SBATCH --job-name=eval_chained
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_chained_%j.out
exec 2>&1

# Generic single-GPU follow-up eval, meant to be submitted with
# --dependency=afterok:<train_job_id> right after a training job (see
# submit_train_then_eval.sh). Reads the wandb run id + saved model path from
# the handoff files the training job wrote via WANDB_RUN_ID_OUTPUT /
# MODEL_PATH_OUTPUT (main.py), so it doesn't need to guess/glob for the
# right timestamped checkpoint dir. Training already does PPL (safe,
# rank0-only, no distributed collective) — this job only needs zero-shot +
# reasoning, hence --skip_ppl.
#
# Usage: sbatch --dependency=afterok:<TRAIN_JOB_ID> \
#          --partition=<PARTITION> --qos=<QOS> \
#          slurm_eval_chained.sh <TRAIN_JOB_ID> <WANDB_PROJECT> <METHOD> <SPARSITY>

TRAIN_JOB_ID=${1:?"Usage: slurm_eval_chained.sh <TRAIN_JOB_ID> <WANDB_PROJECT> <METHOD> <SPARSITY>"}
WANDB_PROJECT=${2:?"Usage: slurm_eval_chained.sh <TRAIN_JOB_ID> <WANDB_PROJECT> <METHOD> <SPARSITY>"}
METHOD=${3:?"Usage: slurm_eval_chained.sh <TRAIN_JOB_ID> <WANDB_PROJECT> <METHOD> <SPARSITY>"}
SPARSITY=${4:?"Usage: slurm_eval_chained.sh <TRAIN_JOB_ID> <WANDB_PROJECT> <METHOD> <SPARSITY>"}

RUN_ID_FILE="/home1/doyoonkim/projects/elsa/logs/handoff_${TRAIN_JOB_ID}_wandb_run_id.txt"
MODEL_PATH_FILE="/home1/doyoonkim/projects/elsa/logs/handoff_${TRAIN_JOB_ID}_model_path.txt"

if [ ! -f "$RUN_ID_FILE" ] || [ ! -f "$MODEL_PATH_FILE" ]; then
    echo "ERROR: handoff file(s) missing for train job ${TRAIN_JOB_ID}:"
    echo "  $RUN_ID_FILE (exists: $([ -f "$RUN_ID_FILE" ] && echo yes || echo no))"
    echo "  $MODEL_PATH_FILE (exists: $([ -f "$MODEL_PATH_FILE" ] && echo yes || echo no))"
    echo "Training job likely failed before finishing — check its own log instead of retrying eval."
    exit 1
fi
WANDB_RUN_ID=$(cat "$RUN_ID_FILE" | tr -d '[:space:]')
MODEL_PATH=$(cat "$MODEL_PATH_FILE" | tr -d '[:space:]')

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false
export WANDB_INIT_TIMEOUT=120

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  TRAIN_JOB=$TRAIN_JOB_ID"
echo "WANDB_RUN_ID=$WANDB_RUN_ID  MODEL_PATH=$MODEL_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_id "$WANDB_RUN_ID" \
    --method "$METHOD" \
    --sparsity ${SPARSITY} \
    --gpu_util 0.85 \
    --tp_size 1 \
    --skip_ppl \
    --out_base /local-data/user-data/${USER}/job_${SLURM_JOB_ID}/eval_out

echo "##### END #####"
