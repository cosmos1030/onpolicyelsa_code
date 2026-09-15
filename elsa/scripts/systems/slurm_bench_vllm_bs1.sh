#!/bin/bash
#SBATCH --job-name=vllm_bs1
# L40S: the same card the HF-side MACKO numbers were taken on. Comparing a
# vLLM run here against an HF run on a different card would answer nothing.
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1
OUTDIR=/home1/doyoonkim/projects/elsa/logs/systems; mkdir -p "$OUTDIR"
LB="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"; mkdir -p "$LB/slurm"
trap 'cp "$LB/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUTDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null || true' EXIT
export TMPDIR=/tmp TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 VLLM_HOST_IP=127.0.0.1
# vLLM's CuMemAllocator asserts against expandable_segments; leave it unset.
echo "NODE=$(hostname) JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /home1/doyoonkim/projects/elsa
/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/systems/bench_vllm_dense_bs1.py \
    --model "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" \
    --out "$OUTDIR/vllm_bs1_${SLURM_JOB_ID}.json"
EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="; echo "##### END #####"; exit $EXIT_CODE
