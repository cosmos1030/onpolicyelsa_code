#!/bin/bash
#SBATCH --job-name=mseval
#SBATCH --partition=A100-80GB,4A100
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
# 16, not 8: this job's whole body is lighteval/vLLM, whose scheduler and
# (de)tokenizer are CPU-bound. The 8-CPU rule is for training jobs -- applying
# it to an eval tail once took SparseGPT runs from ~3h to not finishing in 10.
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/mseval_%x_%j.out
exec 2>&1

# Score one milestone checkpoint from the Hub with the "quick" lighteval profile.
# SLURM port of a100_scripts/eval_milestone_a100.sh from the NHN box, which was
# written for a bare-bash machine with manual CUDA_VISIBLE_DEVICES; here each
# checkpoint gets its own job and SLURM hands it a GPU.
#
# Usage: sbatch --job-name=<short> slurm_eval_milestone.sh <HF_REPO|LOCAL_DIR>
#
# A LOCAL_DIR is scored in place. The 4B milestones came off the Hub because
# they were trained on the NHN box; milestones trained here are already on
# this filesystem, and copying 3.4GB into $WORK just to read it back is pure
# waste -- and worse, it would be a second copy to keep in sync.
#
# tp=1: Qwen3-4B in bf16 plus the 8192-token KV budget fits one 80GB card with
# room to spare, and tp=1 has no cross-GPU collective -- six single-GPU jobs
# backfill far better here than fewer multi-GPU ones.
#
# wandb is deliberately off. The three milestones of one arm would resume into
# the same parent run and overwrite each other's run.summary (avg5,
# math500_pass@1), leaving the trajectory unreadable. Scores come from
# eval_summary_resumed.json.
#
# Note for the results table: these are A100 tp=1, while the 2048 endpoints were
# scored on B200 tp=4. Different hardware and TP path for three of four points
# in each trajectory.

REPO=${1:?"Usage: sbatch slurm_eval_milestone.sh <HF_REPO|LOCAL_DIR>"}
TP=${2:-1}
GPU_UTIL=${3:-0.90}
NAME=$(basename "$REPO")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
ROOT=/home1/doyoonkim/projects
WORK=${WORK:-$ROOT/elsa/logs/milestone_eval}
if [ -f "$REPO/config.json" ]; then
    MODEL_DIR="$REPO"
else
    MODEL_DIR="$WORK/$NAME"
fi
mkdir -p "$WORK"

export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1
export NCCL_DEBUG=WARN
# lighteval pulls its task definitions from the Hub at run time
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

echo "=== milestone eval: $NAME  (tp=$TP gpu_util=$GPU_UTIL) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "=== downloading $REPO ==="
    $PYTHON - "$REPO" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
print("downloaded")
PY
fi

cd "$ROOT/elsa"
$PYTHON "$ROOT/b200_scripts/resume_eval_lighteval.py" \
    --model_dir "$MODEL_DIR" \
    --tp_size "$TP" --gpu_util "$GPU_UTIL" \
    --profile quick --seed 42 --no_hub

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
