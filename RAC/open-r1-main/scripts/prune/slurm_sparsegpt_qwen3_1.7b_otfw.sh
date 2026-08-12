#!/bin/bash
#SBATCH --job-name=sparsegpt_1.7b_otfw
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs_trace/sparsegpt_1.7b_%j.out
exec 2>&1

# SparseGPT one-shot pruning of Qwen3-1.7B, calibrated on the SAME OT80/FW20
# data ALPS uses (elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl), just
# pre-truncated to 2048 tokens/row (data/ot3_fineweb_40k_qwen3_nostrip_2048trunc.jsonl,
# first 400 qualifying rows) -- the original untruncated rows (up to 8192 tok)
# OOM'd eager attention on a 24GB RTX3090 (seq^2 softmax matrix). Chains
# elsa's own eval_full.py afterward so this is directly comparable to
# ALPS/TR-GMP on the same benchmark suite (math500/gpqa/ifeval/lcb/gsm8k +
# zero-shot + PPL), not just RAC's own built-in math500-only eval.
#
# Usage: sbatch slurm_sparsegpt_qwen3_1.7b_otfw.sh <SPARSITY e.g. 0.5>

SPARSITY=${1:?"Usage: sbatch slurm_sparsegpt_qwen3_1.7b_otfw.sh <SPARSITY>"}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

MODEL_HASH="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PRUNED_MODEL="/home1/doyoonkim/projects/RAC/open-r1-main/models/${MODEL_HASH}_pruned_${SPARSITY_PCT}_all_tokens262144_prunemethod_SparseGPT_thirds_1_2_3__ot3_fineweb_40k_qwen3_nostrip_2048trunc.jsonl"

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
LOCAL_JOB_BASE="/local-data/user-data/${USER}/sparsegpt_1.7b_${SLURM_JOB_ID}"
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

echo "=== SparseGPT Qwen3-1.7B s${SPARSITY_PCT}% (OT80/FW20 calib, 2048-trunc, same source as ALPS) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "No internet on $(hostname) -- falling back to WANDB_MODE=offline (sync later)."
    export WANDB_MODE=offline
fi

cd /home1/doyoonkim/projects/RAC/open-r1-main

python src/open_r1/grpo.py \
  --config recipes/Qwen3-1.7B/grpo/config_sparsegpt_otfw.yaml \
  --save_dir /home1/doyoonkim/projects/RAC/open-r1-main/models \
  --report_to wandb \
  --do_train False \
  --prune \
  --pruning_method SparseGPT \
  --prune_sparsity ${SPARSITY} \
  --prune_calib_tokens 262144 \
  --push_to_hub False \
  --score_completions False

PRUNE_EXIT=$?
echo "=== grpo.py prune exit: $PRUNE_EXIT ==="
if [ $PRUNE_EXIT -ne 0 ] || [ ! -d "$PRUNED_MODEL" ]; then
    echo "ERROR: pruned model not found at $PRUNED_MODEL"
    exit 1
fi

echo "=== Running elsa eval_full.py on $PRUNED_MODEL ==="
$PYTHON /home1/doyoonkim/projects/elsa/scripts/eval_full.py \
    --model_path "$PRUNED_MODEL" \
    --wandb_project reasoning_qwen3_1.7b_nostrip8192 \
    --run_name "sparsegpt_s${SPARSITY_PCT}pct" \
    --method sparsegpt \
    --sparsity "$SPARSITY" \
    --gpu_util 0.85 \
    --profile quick \
    --out_base "$LOCAL_JOB_BASE/eval_out"

EVAL_EXIT=$?
echo "=== eval_full.py exit: $EVAL_EXIT ==="
echo "##### END #####"
exit $EVAL_EXIT
