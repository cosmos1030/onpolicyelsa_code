#!/bin/bash
#SBATCH --job-name=r1_trace
#SBATCH --partition=4A100
#SBATCH --qos=4A100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/%j.out
#SBATCH -t 3-00:00:00

# 로그 폴더 생성 (필요 시)
mkdir -p logs

# 1. 고유 ID 생성
ID=$(date +%H%M%S)

# 2. Slurm에서 할당받은 GPU UUID 추출
UUIDLIST=$(nvidia-smi -L | grep UUID | cut -d '(' -f 2 | awk '{print $2}' | tr -d ")" | paste -s -d, -)

# 3. Slurm에서 할당받은 CPU 코어 리스트 추출
CPULIST=$(grep "Cpus_allowed_list" /proc/self/status | awk '{print $2}')

echo "Running Job ID: $SLURM_JOB_ID"
echo "Allocated GPUs: $UUIDLIST"
echo "Allocated CPUs: $CPULIST"

# docker pull dyk6208/open-r1:latest

# 4. Docker 실행 및 내부에서 파이썬 스크립트 실행
docker run --rm \
  --name ${USER}_open_r1_${ID} \
  --gpus '"device='${UUIDLIST}'"' \
  --cpuset-cpus "${CPULIST}" \
  --shm-size=64gb \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  -v $(pwd):/workspace \
  -w /workspace \
  dyk6208/open-r1 \
  python src/open_r1/grpo.py \
    --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_demo.yaml \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset_name ./dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k \
    --output_dir ./checkpoints/pruned_model \
    --beta 0 \
    --report_to wandb \
    --use_vllm False \
    --do_train False \
    --prune \
    --pruning_method SparseGPT \
    --prune_sparsity 0.4 \
    --prune_calib_tokens 1_000_000 \
    --push_to_hub False \
    --score_completions False