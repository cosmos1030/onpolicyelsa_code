#!/bin/bash
#SBATCH --job-name=multisample_s70_full
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/multisample_s70_full_%j.out
exec 2>&1

NFS_LOG_COPY="/home1/doyoonkim/projects/elsa/models/multisample_s70_n8_full_slurmlog_${SLURM_JOB_ID}.out"
mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
copy_log_to_nfs() {
  cp "/local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm/multisample_s70_full_${SLURM_JOB_ID}.out" "$NFS_LOG_COPY" 2>/dev/null
}
trap copy_log_to_nfs EXIT

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

LIGHTEVAL=/home1/doyoonkim/miniconda3/envs/rac/bin/lighteval
PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

echo "=== TR-GMP-s70 full MATH-500, n=8 samples/question, lighteval-native ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

GPU_UTIL=$($PYTHON -c "
import torch
free, total = torch.cuda.mem_get_info(0)
print(f'{free/total*0.90:.4f}')
")
if [ -z "$GPU_UTIL" ]; then
    echo "FATAL: GPU_UTIL empty, torch.cuda.mem_get_info failed"
    exit 1
fi
echo "GPU_UTIL=$GPU_UTIL"

$LIGHTEVAL vllm \
    "model_name=cosmos1030/gmp-kd3e-1-s70pct-lr5e-5_20260811_115604,dtype=bfloat16,trust_remote_code=true,tensor_parallel_size=1,gpu_memory_utilization=${GPU_UTIL},max_model_length=8192,max_num_batched_tokens=8192,seed=42,override_chat_template=true,generation_parameters={max_new_tokens:8192,temperature:0.6,top_p:0.95}" \
    "custom|math_500_n8|0|0" \
    --custom-tasks /home1/doyoonkim/projects/elsa/lib/custom_math500_n8.py \
    --output-dir /home1/doyoonkim/projects/elsa/models/multisample_s70_n8_full \
    --save-details
LIGHTEVAL_EXIT=$?

echo "=== DONE (lighteval exit $LIGHTEVAL_EXIT) ==="
exit $LIGHTEVAL_EXIT
