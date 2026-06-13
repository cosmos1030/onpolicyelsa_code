#!/bin/bash
#SBATCH --job-name=eval_dense
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/%j_eval.out
#SBATCH -t 1-00:00:00

set -euo pipefail
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

MODEL_DIR=/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
MODEL_TAG=DeepSeek-R1-Distill-Qwen-1.5B_dense

MODEL_ARGS="model_name=${MODEL_DIR},\
dtype=bfloat16,\
trust_remote_code=true,\
tensor_parallel_size=4,\
pipeline_parallel_size=1,\
gpu_memory_utilization=0.9,\
max_model_length=32768,\
override_chat_template=true,\
generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

OUTPUT_DIR=/home1/doyoonkim/projects/elsa/eval_results/${MODEL_TAG}

export WANDB_PROJECT=elsa_deepseek_1.5b
export WANDB_NAME=${MODEL_TAG}

lighteval vllm "$MODEL_ARGS" "lighteval|math_500|0|0" \
  --wandb \
  --output-dir "$OUTPUT_DIR" \
  --save-details

echo "Evaluation finished: ${MODEL_TAG}"