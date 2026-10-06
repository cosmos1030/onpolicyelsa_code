#!/bin/bash
#SBATCH --job-name=qwen3_8b_vllm_trace
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/vllm_trace_8b_%j.out
exec 2>&1

# Fast vLLM replacement for the RAC grpo.py --trace_only self-gen trace path
# (which crawled at ~10-27 tok/s with plain HF generate()). Generates exactly
# 150 DISTINCT prompts once each (n=1) -- already "1 answer/problem" like v3,
# no dedup step needed.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Qwen3-8B vLLM self-gen trace (150 distinct prompts, n=1) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

python scripts/vllm_selfgen_trace.py \
  --model_path /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 \
  --prompt_path /home1/doyoonkim/projects/elsa/data/ot3_prompts_2000_qwen3.jsonl \
  --out_path /home1/doyoonkim/projects/elsa/data/selfgen_trace_qwen3_8b_v3_raw.jsonl \
  --n_prompts 150 --seed 42 --max_tokens 8192 --temperature 0.7 --gpu_util 0.9 --tp_size 1

echo "##### END #####"
