#!/bin/bash
#SBATCH --job-name=spinfer_test
#SBATCH --partition=L40S,RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --exclude=n3,n19,n31,n42,n43,n46,n51,n52,n53,n54,n55,n56,n58,n60,n61,n64,n76,n77,n80,n84,n87,n90,n91,n92
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
# SpInfer vLLM plug-in check: op-level correctness (eager + CUDA graph) on real SCOUT weights,
# then greedy batch-8 vLLM generations dense-kernel vs SpInfer on the same sparse model.
exec 2>&1
OUTDIR=/home1/doyoonkim/projects/elsa/logs/systems; mkdir -p "$OUTDIR"
LB="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"; mkdir -p "$LB/slurm"
trap 'cp "$LB/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUTDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null || true' EXIT
export TMPDIR=/tmp TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0 VLLM_NO_USAGE_STATS=1 VLLM_HOST_IP=127.0.0.1
export CUDA_HOME=/opt/ohpc/pub/apps/cuda/12.8
export PATH="/home1/doyoonkim/miniconda3/envs/rac/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "NODE=$(hostname) JOB=$SLURM_JOB_ID"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /home1/doyoonkim/projects/elsa/scripts/systems/macko_vllm && export PYTHONPATH=$PWD:${PYTHONPATH:-}
PY=/home1/doyoonkim/miniconda3/envs/rac/bin/python
CT=/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
O=$LB/ids
$PY test_spinfer.py --model "${MODEL:?MODEL}" --chat_tokenizer $CT || exit 1
SPARSE_BACKEND=spinfer $PY test_spinfer.py --model "$MODEL" --chat_tokenizer $CT --vllm dense --out $O
SPARSE_BACKEND=spinfer MACKO_DROP_DENSE=1 $PY test_spinfer.py --model "$MODEL" --chat_tokenizer $CT --vllm spinfer --out $O
echo "=== EXIT: $? ==="
