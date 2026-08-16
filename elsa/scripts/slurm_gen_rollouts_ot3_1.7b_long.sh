#!/bin/bash
#SBATCH --job-name=gen_rollouts_1.7b_long
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/gen_rollouts_1.7b_long_%j.out
exec 2>&1

# Long (max_new_tokens=16384) OT3 rollout generation for rambling/sequence-
# characteristics analysis, comparing NTP+KD-only (OPD ablation, job 732368)
# vs NTP+KD+OPKD (job 707793) Qwen3-1.7B s70 checkpoints on the SAME 500
# OpenThoughts3 prompts (seed=42). Someone else does the actual analysis --
# this job just produces the two rollout datasets.
#
# Usage: sbatch slurm_gen_rollouts_ot3_1.7b_long.sh <MODEL_PATH_OR_HF_REPO> <OUT_REPO> <OUT_JSONL_NAME>

MODEL_PATH=${1:?"Usage: sbatch slurm_gen_rollouts_ot3_1.7b_long.sh <MODEL_PATH_OR_HF_REPO> <OUT_REPO> <OUT_JSONL_NAME>"}
OUT_REPO=${2:?"missing OUT_REPO"}
OUT_JSONL_NAME=${3:?"missing OUT_JSONL_NAME"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
OUT_JSONL="/home1/doyoonkim/projects/elsa/data/${OUT_JSONL_NAME}"

export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export VLLM_USE_V1=0

echo "=== gen_rollouts_ot3_1.7b_long: model=$MODEL_PATH out_repo=$OUT_REPO ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa/scripts
$PYTHON gen_rollouts_ot3_1.7b_long.py "$MODEL_PATH" "$OUT_REPO" "$OUT_JSONL" 500

echo "=== EXIT: $? ==="
echo "##### END #####"
