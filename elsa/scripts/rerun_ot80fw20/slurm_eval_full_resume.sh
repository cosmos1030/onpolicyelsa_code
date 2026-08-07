#!/bin/bash
#SBATCH --job-name=eval_full_resume
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/eval_full_resume_%j.out
exec 2>&1

# Recovery eval for ELSA plain 4B jobs (689902/689903/689904, submitted 2026-08-02)
# that trained to completion and saved a full checkpoint, but then crashed with
# SIGABRT during zero-shot eval -- root cause: ProcessGroupNCCL watchdog stuck
# for 480s (see project_fsdp_zeroshot_hang.md), an FSDP multi-rank collective
# hang. This script is a standalone SINGLE-PROCESS eval (no FSDP, no process
# group), so the hang shouldn't recur; resumes into the SAME wandb run the
# crashed training job was logging to (--wandb_run_id) so training+eval land
# in one place.
#
# Usage: sbatch slurm_eval_full_resume.sh <MODEL_PATH> <RUN_NAME> <SPARSITY> <WANDB_PROJECT> <WANDB_RUN_ID>
# e.g.: sbatch slurm_eval_full_resume.sh /path/to/model elsa_plain_4b_s60 0.6 reasoning_qwen3_4b xf00weux

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_full_resume.sh <MODEL_PATH> <RUN_NAME> <SPARSITY> <WANDB_PROJECT> <WANDB_RUN_ID>"}
RUN_NAME=${2:-"eval"}
SPARSITY=${3:-0.0}
WANDB_PROJECT=${4:-"reasoning_qwen3_4b"}
WANDB_RUN_ID=${5:?"Usage: sbatch slurm_eval_full_resume.sh <MODEL_PATH> <RUN_NAME> <SPARSITY> <WANDB_PROJECT> <WANDB_RUN_ID>"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"

export WANDB_DIR="/home1/doyoonkim/projects/elsa/logs/wandb_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR"
export WANDB_SERVICE_WAIT=300
export WANDB_START_METHOD=fork
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
if [ -z "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY=$(grep "^export WANDB_API_KEY=" ~/.bashrc 2>/dev/null | cut -d= -f2-)
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_CACHE="/home1/doyoonkim/.cache/huggingface/datasets"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== eval_full_resume: $RUN_NAME (resuming wandb run $WANDB_RUN_ID) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
echo "MODEL_PATH=$MODEL_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_id "$WANDB_RUN_ID" \
    --run_name "$RUN_NAME" \
    --method elsa \
    --sparsity "$SPARSITY" \
    --gpu_util 0.85 \
    --out_base "$LOCAL_JOB_BASE/eval_${RUN_NAME}"

echo "##### END #####"
