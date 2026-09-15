#!/bin/bash
#SBATCH --job-name=macko
# RTX3090 on purpose, not A100. MACKO's README says it is tuned on 2080/3090/
# 4090 and that server GPUs (H100, V100) need separate optimisation; A100 is a
# server GPU. Measuring the format where it is NOT tuned would tell us nothing
# about the format. The consequence is that these numbers cannot share a table
# with the 2:4 results, which were taken on A100.
# A5000/A6000 are the same GA102 silicon as the RTX3090 (sm_86), and
# RTX6000ADA/L40S the same AD102 as the RTX4090 (sm_89), so the kernel is on
# hardware it was tuned for on any of these. The README's warning is about
# H100 (sm_90) and V100 (sm_70), which are genuinely different. Listing all
# five matters in practice: RTX3090 alone had 301 jobs queued.
#SBATCH --partition=L40S
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1

# MACKO-SpMV vs cuBLAS at high unstructured sparsity, on Qwen3 weight shapes.
#
# Usage: sbatch slurm_bench_macko.sh [MODEL_DIR]

MODEL=${1:-/home1/doyoonkim/projects/elsa/models/qwen3_4b_alps_s80pct}

ENVDIR=/home1/doyoonkim/miniconda3/envs/rac
PYTHON=$ENVDIR/bin/python
ROOT=/home1/doyoonkim/projects
OUTDIR=$ROOT/elsa/logs/systems
mkdir -p "$OUTDIR"

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/slurm"
trap 'cp "$LOCAL_JOB_BASE/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUTDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null || true' EXIT

# The kernels are JIT-built by load_inline at import, so this needs nvcc and
# ninja on PATH. CUDA 12.6 to match torch's cu126 build -- the cluster default
# is 13.1, which is a different major version than torch was compiled against.
# 12.8, not 12.6. nvcc checks the host gcc version and 12.6 rejects anything
# newer than 13, while these nodes default to gcc 14.2. Pinning the OHPC gcc
# 13.4.0 instead does not work -- its cc1plus cannot load libisl.so.15, which
# is not present anywhere on the system. CUDA 12.8 accepts gcc 14, and building
# an extension with a 12.8 nvcc against a cu126 torch is fine: CUDA is
# compatible across 12.x minor versions.
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export PATH="$ENVDIR/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# Cache the build per GPU architecture: load_inline compiles for the card it
# sees, so one shared directory would hand an sm_86 binary to a different card.
ARCH=$($PYTHON -c "import torch;print('sm%d%d' % torch.cuda.get_device_capability(0))" 2>/dev/null || echo unknown)
export MACKO_SPMV_BUILD_DIRECTORY=/home1/doyoonkim/.cache/macko_build_${ARCH}
mkdir -p "$MACKO_SPMV_BUILD_DIRECTORY"

export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== MACKO-SpMV benchmark ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  ARCH=$ARCH"
echo "MODEL=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvcc --version | tail -2
ninja --version

cd "$ROOT/elsa"
$PYTHON scripts/systems/bench_macko_spmv.py --model "$MODEL"

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
