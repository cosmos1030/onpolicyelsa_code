#!/bin/bash
#SBATCH --job-name=alps_4b_dyncalib
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/alps_4b_dyncalib_%j.out
exec 2>&1

# Real-scale run of qwen3_alps_dynamic_calib.py: FW20 fixed, OT80 (102
# slots) with refresh_ratio=0.25 of them replaced via self-gen (using the
# model as pruned through the current layer) at every one of the 36 decoder
# layer boundaries. seqlen=8192, nsamples=128 -- matches static ALPS's own
# defaults exactly, for a fair comparison against the static OT80/FW20 and
# self-gen-v2 baselines.
#
# Usage: sbatch slurm_alps_dynamic_calib_4b.sh <SPARSITY>

SPARSITY=${1:?"Usage: sbatch slurm_alps_dynamic_calib_4b.sh <SPARSITY>"}
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
SAVE="/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_dyncalib_s${SPARSITY_PCT}pct"
OUT_LOG="/home1/doyoonkim/projects/ALPS/kldiag_out/dyncalib_4b_s${SPARSITY_PCT}pct.jsonl"

export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export VLLM_USE_V1=0

echo "=== ALPS dynamic-calib pruning Qwen3-4B s${SPARSITY_PCT}pct (refresh_ratio=0.25, seqlen=8192) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "WARNING: No internet on $(hostname); push_to_hub will fail but pruning/save will proceed."
fi

cd /home1/doyoonkim/projects/ALPS

$PYTHON qwen3_alps_dynamic_calib.py \
    "$MODEL" ${SPARSITY} \
    --seqlen 8192 \
    --n_ot 102 --n_fw 26 \
    --refresh_ratio 0.25 \
    --gen_max_new_tokens 8192 --vllm_gpu_mem 0.4 --vllm_max_prompt_len 1024 \
    --rho 300.0 \
    --seed 42 \
    --save "$SAVE" \
    --out "$OUT_LOG" \
    --push_to_hub \
    --hub_model_id "cosmos1030/alps-dyncalib-s${SPARSITY_PCT}pct"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
