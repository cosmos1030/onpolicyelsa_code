#!/bin/bash
#SBATCH --job-name=verify_n8
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/verify_n8_%j.out
exec 2>&1

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

LIGHTEVAL=/home1/doyoonkim/miniconda3/envs/rac/bin/lighteval
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

GPU_UTIL=$($PYTHON -c "
import torch
free, total = torch.cuda.mem_get_info(0)
print(f'{free/total*0.90:.4f}')
")

$LIGHTEVAL vllm \
    "model_name=cosmos1030/gmp-kd3e-1-s70pct-lr5e-5_20260811_115604,dtype=bfloat16,trust_remote_code=true,tensor_parallel_size=1,gpu_memory_utilization=${GPU_UTIL},max_model_length=8192,max_num_batched_tokens=8192,seed=42,override_chat_template=true,generation_parameters={max_new_tokens:8192,temperature:0.6,top_p:0.95}" \
    "custom|math_500_n8|0|0" \
    --custom-tasks /home1/doyoonkim/projects/elsa/lib/custom_math500_n8.py \
    --max-samples 5 \
    --output-dir /home1/doyoonkim/projects/elsa/models/_verify_n8 \
    --save-details

echo "=== DONE ==="
