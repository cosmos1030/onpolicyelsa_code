#!/bin/bash
#SBATCH --job-name=math500_savedetails_4b
#SBATCH --partition=RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/math500_sd_%j.out
exec 2>&1

# Usage:
#   sbatch slurm_eval_math500_savedetails_4b.sh <MODEL_PATH_OR_HF_REPO> <OUTPUT_DIR>
#
# Runs MATH-500 only, with --save-details, matching the 8192-budget protocol
# used across the qwen3_4b_nostrip8192 reasoning-bench sweeps (temp=0.6, top_p=0.95,
# max_new_tokens=8192). Purpose: get a fresh GENERATIVE.parquet (output_tokens +
# gradeable text) for a specific checkpoint whose local eval artifacts were lost.

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_math500_savedetails_4b.sh <MODEL_PATH_OR_HF_REPO> <OUTPUT_DIR>"}
OUTPUT_DIR=${2:?"Usage: sbatch slurm_eval_math500_savedetails_4b.sh <MODEL_PATH_OR_HF_REPO> <OUTPUT_DIR>"}

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
NFS_LOG_COPY="${2:-/home1/doyoonkim/projects/elsa/models}/slurmlog_${SLURM_JOB_ID}.out"
copy_log_to_nfs() {
  cp "/local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm/math500_sd_${SLURM_JOB_ID}.out" "$NFS_LOG_COPY" 2>/dev/null
}
trap copy_log_to_nfs EXIT
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

LIGHTEVAL=/home1/doyoonkim/miniconda3/envs/rac/bin/lighteval
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

echo "=== math500_savedetails_4b ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

GPU_UTIL=$($PYTHON -c "
import torch
free, total = torch.cuda.mem_get_info(0)
print(f'{free/total*0.92:.4f}')
")
if [ -z "$GPU_UTIL" ]; then
    echo "FATAL: GPU_UTIL empty, torch.cuda.mem_get_info failed"
    exit 1
fi
echo "GPU_UTIL=$GPU_UTIL"

mkdir -p "$OUTPUT_DIR"

$LIGHTEVAL vllm \
    "model_name=${MODEL_PATH},dtype=bfloat16,trust_remote_code=true,tensor_parallel_size=1,gpu_memory_utilization=${GPU_UTIL},max_model_length=8192,max_num_batched_tokens=8192,seed=42,override_chat_template=true,generation_parameters={max_new_tokens:8192,temperature:0.6,top_p:0.95}" \
    "lighteval|math_500|0|0" \
    --output-dir "$OUTPUT_DIR" \
    --save-details
LIGHTEVAL_EXIT=$?

echo "=== DONE (lighteval exit $LIGHTEVAL_EXIT) ==="
exit $LIGHTEVAL_EXIT
