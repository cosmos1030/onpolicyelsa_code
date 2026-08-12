#!/bin/bash
#SBATCH --job-name=eval_sparsegpt_s50
#SBATCH --partition=RTX3090,RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs_trace/eval_sparsegpt_s50_%j.out
exec 2>&1

# eval_full.py-only run for the s50 SparseGPT checkpoints already pruned by
# the canary jobs (716781 for 1.7B, 716785 for 4B) -- avoids re-pruning.
#
# Usage: sbatch slurm_eval_sparsegpt_s50.sh <1.7b|4b>

SIZE=${1:?"Usage: sbatch slurm_eval_sparsegpt_s50.sh <1.7b|4b>"}
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
LOCAL_JOB_BASE="/local-data/user-data/${USER}/eval_sparsegpt_s50_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/eval_out"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

if [ "$SIZE" = "1.7b" ]; then
    MODEL_PATH="/home1/doyoonkim/projects/RAC/open-r1-main/models/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e_pruned_50_all_tokens262144_prunemethod_SparseGPT_thirds_1_2_3__ot3_fineweb_40k_qwen3_nostrip_2048trunc.jsonl"
    WANDB_PROJECT="reasoning_qwen3_1.7b_nostrip8192"
else
    MODEL_PATH="/home1/doyoonkim/projects/RAC/open-r1-main/models/1cfa9a7208912126459214e8b04321603b3df60c_pruned_50_all_tokens262144_prunemethod_SparseGPT_thirds_1_2_3__ot3_fineweb_40k_qwen3_nostrip_2048trunc.jsonl"
    WANDB_PROJECT="reasoning_qwen3_4b_nostrip8192"
fi

echo "=== eval_full.py for SparseGPT ${SIZE} s50 checkpoint ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$MODEL_PATH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "No internet on $(hostname) -- falling back to WANDB_MODE=offline (sync later)."
    export WANDB_MODE=offline
fi

$PYTHON /home1/doyoonkim/projects/elsa/scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "sparsegpt_s50pct" \
    --method sparsegpt \
    --sparsity 0.5 \
    --gpu_util 0.85 \
    --profile quick \
    --out_base "$LOCAL_JOB_BASE/eval_out"

EVAL_EXIT=$?
echo "=== eval_full.py exit: $EVAL_EXIT ==="
echo "##### END #####"
exit $EVAL_EXIT
