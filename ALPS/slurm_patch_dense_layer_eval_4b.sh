#!/bin/bash
#SBATCH --job-name=alps_4b_denseL_eval
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/alps_4b_denseL_eval_%j.out
exec 2>&1

# Cheap ablation for the KL-diagnostic finding (kldiag_out/qwen3_4b_s*.jsonl):
# L35's mlp.up_proj is consistently the single largest incremental-KL spike
# across s50/s60/s70, growing ~5x from s50 to s70. Instead of re-running ALPS
# from scratch, just patch that whole decoder layer's 7 projections back to
# their original dense weights inside the ALREADY-PRUNED checkpoint, then run
# the same eval_full.py used for the baseline ALPS entries -- if the spike is
# really the bottleneck, this should measurably move math500/PPL versus the
# untouched s70 baseline (job 719640-equivalent local sweep: math500=35.8).
#
# Usage: sbatch slurm_patch_dense_layer_eval_4b.sh <SPARSITY_PCT e.g. 70> <LAYER_IDX e.g. 35>

SPARSITY_PCT=${1:?"Usage: sbatch slurm_patch_dense_layer_eval_4b.sh <SPARSITY_PCT> <LAYER_IDX>"}
LAYER_IDX=${2:?"Usage: sbatch slurm_patch_dense_layer_eval_4b.sh <SPARSITY_PCT> <LAYER_IDX>"}
SPARSITY=$(python3 -c "print(${SPARSITY_PCT}/100)")

PRUNED_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_s${SPARSITY_PCT}pct"
DENSE_MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
PATCHED_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_s${SPARSITY_PCT}pct_denseL${LAYER_IDX}"

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
LOCAL_JOB_BASE="/local-data/user-data/${USER}/alps_4b_denseL_eval_${SLURM_JOB_ID}"
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

echo "=== Patch layer ${LAYER_IDX} to dense in ALPS s${SPARSITY_PCT}pct, then eval ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/ALPS

$PYTHON patch_dense_layer.py "$PRUNED_MODEL" "$DENSE_MODEL" ${LAYER_IDX} --save "$PATCHED_MODEL"
PATCH_EXIT=$?
if [ $PATCH_EXIT -ne 0 ]; then
    echo "ERROR: patch_dense_layer.py failed with exit $PATCH_EXIT" >&2
    exit 1
fi

echo "=== Running eval_full.py on $PATCHED_MODEL ==="
$PYTHON /home1/doyoonkim/projects/elsa/scripts/eval_full.py \
    --model_path "$PATCHED_MODEL" \
    --wandb_project reasoning_qwen3_4b \
    --run_name "alps_s${SPARSITY_PCT}pct_denseL${LAYER_IDX}" \
    --method alps \
    --sparsity ${SPARSITY} \
    --gpu_util 0.90 \
    --profile quick \
    --out_base "$LOCAL_JOB_BASE/eval_out"

EVAL_EXIT=$?
echo "=== eval_full.py exit: $EVAL_EXIT ==="
echo "##### END #####"
