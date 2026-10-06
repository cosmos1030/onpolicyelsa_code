#!/bin/bash
#SBATCH --job-name=reeval_gsm8k_s70
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91,n87,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/reeval_gsm8k_s70_%j.out
exec 2>&1

# job 825468 (SparseLLM Qwen3-1.7B s70%, wandb rcqvn3qf) crashed mid-gsm8k
# with RecursionError (str() on a deeply-nested sympy expr from the
# heavily-degraded S70 model's garbage output -- see the
# sys.setrecursionlimit(10000) fix just added to
# lighteval_patched_runner.py). Model checkpoint + all other benchmarks
# (math500/gpqa/ifeval/lcb) + zero-shot + PPL already completed and logged
# to rcqvn3qf; this just reruns gsm8k standalone and appends to that SAME
# wandb run so nothing else needs to be redone.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL_PATH=/home1/doyoonkim/projects/elsa/models/qwen3_1.7b_sparsellm_s70pct

export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)

echo "=== Re-eval gsm8k only for job 825468 checkpoint (RecursionError fix applied), resuming wandb rcqvn3qf ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$MODEL_PATH" \
    --wandb_project reasoning_qwen3_1.7b \
    --wandb_run_id rcqvn3qf \
    --method sparsellm \
    --sparsity 0.7 \
    --gpu_util 0.9 \
    --tp_size 1 \
    --profile quick \
    --benchmarks gsm8k \
    --skip_ppl \
    --skip_zeroshot \
    --out_base /local-data/user-data/${USER}/reeval_gsm8k_s70_${SLURM_JOB_ID}/eval_out

EXIT_CODE=$?
echo "=== eval_full.py EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
