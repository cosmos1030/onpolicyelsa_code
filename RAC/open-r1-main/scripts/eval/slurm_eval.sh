#!/bin/bash
#SBATCH --job-name=r1_eval
#SBATCH --partition=4A100
#SBATCH --qos=4A100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=logs/%j.out
#SBATCH -t 1-00:00:00

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

COMMAND="lighteval vllm model_name=./models/DeepSeek-R1-Distill-Qwen-1_pruned_40_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k,dtype=bfloat16,trust_remote_code=true,max_model_length=32768,gpu_memory_utilization=0.8,data_parallel_size=4,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95} lighteval|math_500|0|0 --use-chat-template --output-dir ./eval_results/DeepSeek-R1-Distill-Qwen-1_pruned_40_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k --save-details"

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
  /bin/bash -c "$COMMAND"