#!/bin/bash
#SBATCH --job-name=alps_4b_dyncalib_eval
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/alps_4b_dyncalib_eval_%j.out
exec 2>&1

# qwen3_alps_dynamic_calib.py itself only measures its own in-domain
# get_ot_fw PPL (not comparable to other rows' wt2/c4) and never runs the
# lighteval reasoning suite. Run eval_full.py separately on the saved
# checkpoint for a real, comparable math500/gpqa/ifeval/lcb/gsm8k readout.
#
# Usage: sbatch slurm_eval_dyncalib_4b.sh <SPARSITY_PCT e.g. 50>

SPARSITY_PCT=${1:?"Usage: sbatch slurm_eval_dyncalib_4b.sh <SPARSITY_PCT>"}
SPARSITY=$(python3 -c "print(${SPARSITY_PCT}/100)")

MODEL_PATH="/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_dyncalib_s${SPARSITY_PCT}pct"
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
LOCAL_JOB_BASE="/local-data/user-data/${USER}/alps_4b_dyncalib_eval_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/eval_out"

export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

echo "=== eval_full.py on dyncalib s${SPARSITY_PCT}pct ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PYTHON /home1/doyoonkim/projects/elsa/scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project reasoning_qwen3_4b \
    --run_name "alps_dyncalib_s${SPARSITY_PCT}pct" \
    --method alps \
    --sparsity ${SPARSITY} \
    --gpu_util 0.90 \
    --profile quick \
    --out_base "$LOCAL_JOB_BASE/eval_out"

echo "=== eval_full.py exit: $? ==="
echo "##### END #####"
