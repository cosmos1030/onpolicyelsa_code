#!/bin/bash
#SBATCH --job-name=alps_kldiag_4b
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/alps_kldiag_4b_%j.out
exec 2>&1

# Layer-wise KL diagnostic for ALPS pruning of Qwen3-4B: for every individual
# Linear pruned (ALPS's own processing granularity), log KL(dense||current)
# and KL(prev||current) via a full-model forward on a small held-out batch,
# plus that Linear's own ALPS reconstruction error. Tests whether functional
# damage concentrates in specific layers at high sparsity (spiky incremental
# KL) or just accumulates smoothly (flat incremental KL, rising cumulative
# KL) -- see qwen3_alps_kldiag.py docstring. Light on GPU: whole 4B model
# stays resident in bf16 (~8GB) plus small held-out-batch forward passes, no
# training/backward at all.
#
# Usage: sbatch slurm_alps_kldiag_qwen3_4b.sh <SPARSITY e.g. 0.5>

SPARSITY=${1:?"Usage: sbatch slurm_alps_kldiag_qwen3_4b.sh <SPARSITY>"}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
OUT="/home1/doyoonkim/projects/ALPS/kldiag_out/qwen3_4b_s${SPARSITY_PCT}pct.jsonl"
mkdir -p /home1/doyoonkim/projects/ALPS/kldiag_out

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export CUDA_LAUNCH_BLOCKING=1

echo "=== ALPS KL diagnostic: Qwen3-4B s${SPARSITY_PCT}% ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/ALPS

$PYTHON qwen3_alps_kldiag.py "$MODEL" ${SPARSITY} \
    --data_path "$DATA_PATH" \
    --nsamples 32 \
    --heldout 4 \
    --seqlen 1024 \
    --seed 42 \
    --out "$OUT"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
