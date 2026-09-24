#!/bin/bash
#SBATCH --job-name=vllm_macko_8b
# L40S: the same card the HF-side MACKO numbers were taken on. Comparing a
# vLLM run here against an HF run on a different card would answer nothing.
#SBATCH --partition=RTX6000ADA,L40S,A6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
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
echo "NODE=$(hostname) JOB=$SLURM_JOB_ID TAG=${TAG:-s70}"
echo "SPARSE=${SPARSE_MODEL:-(기본 s70)}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /home1/doyoonkim/projects/elsa
# MACKO 커널은 import 시점에 JIT 빌드된다: nvcc(gcc 14 허용하는 12.8)와 ninja 필요.
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export PATH="/home1/doyoonkim/miniconda3/envs/rac/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
ARCH=$(/home1/doyoonkim/miniconda3/envs/rac/bin/python -c "import torch;print('sm%d%d' % torch.cuda.get_device_capability(0))")
export MACKO_SPMV_BUILD_DIRECTORY=/home1/doyoonkim/.cache/macko_build_${ARCH}
mkdir -p "$MACKO_SPMV_BUILD_DIRECTORY"

/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/systems/macko_vllm/bench_vllm_macko.py \
    --dense "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218" \
    --sparse "${SPARSE_MODEL:-/home1/doyoonkim/projects/elsa/models/qwen3_8b_selfgen_v3_alps_s70pct}" \
    --gpu_util 0.80 \
    --out "$OUTDIR/vllm_macko_8b_${TAG:-s70}_${SLURM_JOB_ID}.json"
EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="; echo "##### END #####"; exit $EXIT_CODE
