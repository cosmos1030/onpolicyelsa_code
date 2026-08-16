#!/bin/bash
#SBATCH --job-name=alps_4b_L35sp0_eval
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/alps_4b_L35sp0_eval_%j.out
exec 2>&1

# Follow-up to slurm_patch_dense_layer_eval_4b.sh's L35-dense-patch ablation
# (729531, math500 40.2/gpqa 6.6 vs untouched s70 baseline 35.8/25.8 --
# ambiguous, gpqa cratered). That experiment skipped ALPS's own math for
# layer 35 entirely (raw dense weights copied in). This one instead re-runs
# ALPS's Hessian-collection + ADMM reconstruction on layer 35 with sp=0.0 --
# same calibration data/seed as the original run, cheap because layers 0-34
# don't depend on layer 35's state at all (see patch_alps_layer_sp0.py's
# docstring) -- isolating whether ALPS's reconstruction math itself perturbs
# a layer even at zero sparsity, vs. patch_dense_layer.py's "skip ALPS
# entirely" control.
#
# Usage: sbatch slurm_patch_alps_layer_sp0_eval_4b.sh <SPARSITY_PCT e.g. 70> <LAYER_IDX e.g. 35>

SPARSITY_PCT=${1:?"Usage: sbatch slurm_patch_alps_layer_sp0_eval_4b.sh <SPARSITY_PCT> <LAYER_IDX>"}
LAYER_IDX=${2:?"Usage: sbatch slurm_patch_alps_layer_sp0_eval_4b.sh <SPARSITY_PCT> <LAYER_IDX>"}
SPARSITY=$(python3 -c "print(${SPARSITY_PCT}/100)")

PRUNED_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_s${SPARSITY_PCT}pct"
PATCHED_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_s${SPARSITY_PCT}pct_L${LAYER_IDX}sp0"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
LOCAL_JOB_BASE="/local-data/user-data/${USER}/alps_4b_L35sp0_eval_${SLURM_JOB_ID}"
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

echo "=== Re-run ALPS math (sp=0.0) on layer ${LAYER_IDX} in ALPS s${SPARSITY_PCT}pct, then eval ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/ALPS

$PYTHON patch_alps_layer_sp0.py "$PRUNED_MODEL" ${LAYER_IDX} \
    --data_path "$DATA_PATH" \
    --nsamples 128 --seed 42 --seqlen 2048 --rho 300.0 \
    --save "$PATCHED_MODEL"
PATCH_EXIT=$?
if [ $PATCH_EXIT -ne 0 ]; then
    echo "ERROR: patch_alps_layer_sp0.py failed with exit $PATCH_EXIT" >&2
    exit 1
fi

echo "=== Running eval_full.py on $PATCHED_MODEL ==="
$PYTHON /home1/doyoonkim/projects/elsa/scripts/eval_full.py \
    --model_path "$PATCHED_MODEL" \
    --wandb_project reasoning_qwen3_4b \
    --run_name "alps_s${SPARSITY_PCT}pct_L${LAYER_IDX}sp0" \
    --method alps \
    --sparsity ${SPARSITY} \
    --gpu_util 0.90 \
    --profile quick \
    --out_base "$LOCAL_JOB_BASE/eval_out"

EVAL_EXIT=$?
echo "=== eval_full.py exit: $EVAL_EXIT ==="
echo "##### END #####"
