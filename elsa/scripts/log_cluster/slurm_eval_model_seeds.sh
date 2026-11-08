#!/bin/bash
#SBATCH --job-name=eval_seeds
#SBATCH --partition=A6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/doyoonkim/onpolicyelsa_code/elsa/logs/eval_seeds_%j.out
exec 2>&1

# Standalone lighteval (long profile, TP=1) for an already-trained model, logged
# into its training wandb run. Same evaluator as the in-job eval
# (scripts/eval_full.py -> lib/lighteval_bench.py, patched runner).
#   SEEDS="42"     -> plain keys (lighteval/<bench>), e.g. to fill a benchmark the
#                     in-job eval lost
#   SEEDS="0,1"    -> lighteval/<bench>_seed0/_seed1 + _mean/_std
#
# Usage: sbatch [-p A6000|3090] slurm_eval_model_seeds.sh <MODEL_PATH> <WANDB_PROJECT> <WANDB_RUN_ID> <BENCHMARKS> <SEEDS>
# e.g.:  sbatch slurm_eval_model_seeds.sh /home/.../models/gmp_8b_... reasoning_qwen3_8b_nostrip8192 c5jhns66 math500 42

MODEL_PATH=${1:?"Usage: <MODEL_PATH> <WANDB_PROJECT> <WANDB_RUN_ID> <BENCHMARKS> <SEEDS>"}
WANDB_PROJECT=${2:?}
WANDB_RUN_ID=${3:?}
BENCHMARKS=${4:?}
SEEDS=${5:?}
REPO_ROOT="/home/doyoonkim/onpolicyelsa_code/elsa"
OUT_BASE="/tmp/${USER}/eval_$(basename "$MODEL_PATH")_${SLURM_JOB_ID}"

if [ ! -f "$MODEL_PATH/config.json" ]; then echo "ERROR: no model at $MODEL_PATH" >&2; exit 1; fi

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=${WANDB_API_KEY:-$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)}
export HF_HOME=/home/shared/huggingface
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_HOST_IP=127.0.0.1
export VLLM_USE_V1=0
export TMPDIR=/tmp

echo "=== eval $(basename "$MODEL_PATH") benchmarks=$BENCHMARKS seeds=$SEEDS -> wandb $WANDB_PROJECT/$WANDB_RUN_ID ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$REPO_ROOT"
if [[ "$SEEDS" == *,* ]]; then SEED_ARGS="--seeds=$SEEDS"; else SEED_ARGS="--seed=$SEEDS"; fi
python scripts/eval_full.py \
    --model_path="$MODEL_PATH" \
    --wandb_project="$WANDB_PROJECT" \
    --wandb_run_id="$WANDB_RUN_ID" \
    --skip_ppl \
    --skip_zeroshot \
    --out_base="$OUT_BASE" \
    --gpu_util=0.85 \
    --tp_size=1 \
    --profile=long \
    --benchmarks="$BENCHMARKS" \
    $SEED_ARGS

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
