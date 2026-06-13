#!/bin/bash
#SBATCH --job-name=r1_trace_vllm
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/math_trace_%j.out
#SBATCH -t 3-00:00:00

cd /home1/doyoonkim/projects/RAC/open-r1-main
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

mkdir -p logs

# Start TRL vllm server in background (has /get_world_size endpoint required by GRPOTrainer)
python src/open_r1/open_r1_trl/trl/scripts/vllm_serve.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --port 8000 \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.5 \
  --enforce_eager False &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vllm server..."
for i in $(seq 1 60); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "vllm server ready"
    break
  fi
  sleep 5
done

python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --dataset_name open-r1/OpenR1-Math-220k \
  --dataset_prompt_column problem \
  --save_dir /home1/doyoonkim/projects/RAC/open-r1-main/math_trace_10M \
  --num_generations 2 \
  --beta 0 \
  --report_to wandb \
  --use_vllm True \
  --vllm_server_host http://localhost:8000 \
  --max_completion_length 8192 \
  --do_train False \
  --trace_only \
  --trace_tokens 10000000
