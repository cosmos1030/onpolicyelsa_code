#!/bin/bash
#SBATCH --job-name=macko_bench
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/macko_bench_%j.out
#SBATCH --exclude=n3,n42,n51,n52,n54,n55,n58,n60,n76,n77,n80
exec 2>&1

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

SPARSE_MODEL=/home1/doyoonkim/projects/elsa/models/gmp_s70pct_tropkd_dense_kl002_89hbf71t
DENSE_MODEL=/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
MACKO_DIR=/home1/doyoonkim/projects/macko_spmv
VENV=/tmp/macko_venv_${SLURM_JOB_ID}

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_HOME=/home1/doyoonkim/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_OFFLINE
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp
export TRITON_CACHE_DIR=/tmp/triton_cache_macko_${SLURM_JOB_ID}

# ── Build isolated venv with torch 2.8.0 ─────────────────────────────────────
echo "=== Setting up venv ==="
python3.12 -m venv "$VENV" || /home1/doyoonkim/miniconda3/bin/python3.12 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install --quiet transformers==4.55.0 accelerate

echo "=== Installing macko_spmv ==="
pip install --quiet "$MACKO_DIR"

echo "=== torch version ==="
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'device:', torch.cuda.get_device_name(0))"

echo "=== macko_spmv import test ==="
python -c "import macko_spmv; print('macko_spmv OK')"

# ── Run benchmark ─────────────────────────────────────────────────────────────
echo "=== Starting benchmark ==="
python "$MACKO_DIR/benchmark_qwen3.py" \
    --sparse_model "$SPARSE_MODEL" \
    --dense_model  "$DENSE_MODEL" \
    --dtype float16

echo "##### END #####"
