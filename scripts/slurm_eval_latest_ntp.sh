#!/bin/bash
#SBATCH --job-name=eval_ntp_c4
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

# Find most recently saved NTP-ADMM model (non-kd)
MODEL_DIR=$(ls -dt /home1/doyoonkim/projects/elsa/models/DeepSeek-R1-Distill-Qwen-1.5B_pruned0.5_admm_lr* 2>/dev/null | grep -v kd_admm | head -1)
echo "Evaluating: ${MODEL_DIR}"
MODEL_TAG=$(basename "$MODEL_DIR")

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
