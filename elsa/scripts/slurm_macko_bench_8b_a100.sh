#!/bin/bash
#SBATCH --job-name=macko_8b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/macko_bench_8b_%j.out
#SBATCH --exclude=n3,n42,n51,n52,n54,n55,n58,n60,n76,n77,n80
exec 2>&1

module load cuda/12.8

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

SPARSE_MODEL=/home1/doyoonkim/projects/elsa/models/qwen3_8b_sgpt_s70pct_n128
DENSE_MODEL=/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
PYTHON=/home1/doyoonkim/miniconda3/envs/macko/bin/python
MACKO_DIR=/home1/doyoonkim/projects/macko_spmv

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME=/home1/doyoonkim/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp
export TRITON_CACHE_DIR=/tmp/triton_cache_macko_${SLURM_JOB_ID}

echo "=== Python / torch version ==="
$PYTHON -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'device:', torch.cuda.get_device_name(0))"

echo "=== macko_spmv import test ==="
$PYTHON -c "import macko_spmv; print('macko_spmv OK')"

echo "=== Starting benchmark: Qwen3-8B SparseGPT S70 vs Dense ==="
$PYTHON "$MACKO_DIR/benchmark_qwen3.py" \
    --sparse_model "$SPARSE_MODEL" \
    --dense_model  "$DENSE_MODEL" \
    --dtype float16

echo "##### END #####"
