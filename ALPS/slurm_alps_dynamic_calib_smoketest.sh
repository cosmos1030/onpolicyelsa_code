#!/bin/bash
#SBATCH --job-name=alps_dyncalib_smoketest
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/alps_dyncalib_smoketest_%j.out
exec 2>&1

# Correctness-only smoke test for qwen3_alps_dynamic_calib.py (the new
# EMA-style dynamic OT calibration scheme) BEFORE committing to a real
# 4B/s50-s70 run. Deliberately tiny/fast (seqlen=512, nsamples=8,
# max_layers=2, 32-token generations) -- only checking the pipeline runs
# end to end (capture_inps_for_layer hook, ALPS pruning, pool
# regeneration via self-gen, packing wraparound) without crashing.
# NOT representative of real pruning quality; the real run uses seqlen=8192.

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export VLLM_USE_V1=0

echo "=== Dynamic-calib ALPS smoke test ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/ALPS

$PYTHON qwen3_alps_dynamic_calib.py \
    "$MODEL" 0.5 \
    --seqlen 256 \
    --n_ot 8 --n_fw 2 --refresh_ratio 0.25 \
    --gen_max_new_tokens 64 --oversample 4 \
    --vllm_gpu_mem 0.5 --vllm_max_prompt_len 2048 \
    --max_layers 2 \
    --skip_eval \
    --seed 0 \
    --out /home1/doyoonkim/projects/ALPS/kldiag_out/dynamic_calib_smoketest.jsonl

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
