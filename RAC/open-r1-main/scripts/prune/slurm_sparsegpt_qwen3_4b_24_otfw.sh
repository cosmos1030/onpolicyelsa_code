#!/bin/bash
#SBATCH --job-name=sparsegpt_4b_24_otfw
#SBATCH --partition=RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs_trace/sparsegpt_4b_24_%j.out
exec 2>&1

# Same as slurm_sparsegpt_qwen3_4b_otfw.sh but 2:4 semi-structured
# (--prune_N 2 --prune_M 4), matching the ALPS/TR-GMP 2:4 comparison points.
# grpo.py's own save-path resolution appends an "N2_M4" tag automatically
# (see grpo.py's nm_tag logic), so we glob for the resulting directory
# instead of hardcoding the exact path.

MODEL_HASH="1cfa9a7208912126459214e8b04321603b3df60c"
SAVE_DIR="/home1/doyoonkim/projects/RAC/open-r1-main/models"

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
LOCAL_JOB_BASE="/local-data/user-data/${USER}/sparsegpt_4b_24_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/eval_out"
mkdir -p logs_trace

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

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

echo "=== SparseGPT Qwen3-4B 2:4 semi-structured (OT80/FW20 calib, 2048-trunc, same source as ALPS) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "No internet on $(hostname) -- falling back to WANDB_MODE=offline (sync later)."
    export WANDB_MODE=offline
fi

cd /home1/doyoonkim/projects/RAC/open-r1-main

python src/open_r1/grpo.py \
  --config recipes/Qwen3-4B/grpo/config_sparsegpt_otfw.yaml \
  --save_dir "$SAVE_DIR" \
  --report_to wandb \
  --do_train False \
  --prune \
  --pruning_method SparseGPT \
  --prune_sparsity 0.5 \
  --prune_N 2 \
  --prune_M 4 \
  --prune_calib_tokens 262144 \
  --push_to_hub False \
  --score_completions False

PRUNE_EXIT=$?
echo "=== grpo.py prune exit: $PRUNE_EXIT ==="

PRUNED_MODEL=$(ls -dt "$SAVE_DIR"/${MODEL_HASH}_pruned_50_*N2_M4*ot3_fineweb_40k_qwen3_nostrip_2048trunc.jsonl 2>/dev/null | head -1)
if [ $PRUNE_EXIT -ne 0 ] || [ -z "$PRUNED_MODEL" ] || [ ! -d "$PRUNED_MODEL" ]; then
    echo "ERROR: pruned model not found (glob for ${MODEL_HASH}_pruned_50_*N2_M4*)"
    exit 1
fi
echo "PRUNED_MODEL=$PRUNED_MODEL"

echo "=== Running elsa eval_full.py on $PRUNED_MODEL ==="
$PYTHON /home1/doyoonkim/projects/elsa/scripts/eval_full.py \
    --model_path "$PRUNED_MODEL" \
    --wandb_project reasoning_qwen3_4b_nostrip8192 \
    --run_name "sparsegpt_s24" \
    --method sparsegpt \
    --sparsity 0.5 \
    --gpu_util 0.85 \
    --profile quick \
    --out_base "$LOCAL_JOB_BASE/eval_out"

EVAL_EXIT=$?
echo "=== eval_full.py exit: $EVAL_EXIT ==="
echo "##### END #####"
exit $EVAL_EXIT
