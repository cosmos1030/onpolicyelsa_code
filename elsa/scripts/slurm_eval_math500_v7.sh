#!/bin/bash
#SBATCH --job-name=eval_v7
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --exclude=n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/eval_v7_%j.out
exec 2>&1

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb

export WANDB_DIR=/local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1 | tr -d ' \n\r')
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0

MODEL_PATH=$1   # path to saved sparse model
WANDB_RUN_ID=$2 # wandb run id to log to (optional)

echo "=== math500 eval: $MODEL_PATH ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<PYEOF
import os, sys, torch
sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
from lib.lighteval_math500 import run_lighteval_math500
import wandb

model_path = '$MODEL_PATH'
free_mem, total_mem = torch.cuda.mem_get_info(0)
gpu_util = (free_mem / total_mem) * 0.95
print(f'vLLM gpu_memory_utilization: {gpu_util:.3f}')

pass_at_1 = run_lighteval_math500(
    model_path=model_path,
    output_dir=os.path.join(model_path, 'lighteval_math500'),
    max_new_tokens=8192,
    max_samples=500,
    tensor_parallel_size=1,
    gpu_memory_utilization=gpu_util,
    seed=42,
    log_to_wandb=False,
)
print(f'math500_pass@1 = {pass_at_1:.4f}')
PYEOF

echo "=== EVAL DONE ==="
