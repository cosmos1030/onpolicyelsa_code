#!/bin/bash
#SBATCH --job-name=debug_cuda
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs/debug_cuda_%j.out
#SBATCH -t 0-00:10:00

echo "=== SLURM ENV ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "HOSTNAME=$(hostname)"

echo ""
echo "=== ENV_FILE ==="
ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
echo "ENV_FILE path: $ENV_FILE"
if [ -f "$ENV_FILE" ]; then
    echo "ENV_FILE exists:"
    cat "$ENV_FILE"
else
    echo "ENV_FILE does not exist"
fi

echo ""
echo "=== After sourcing ENV_FILE ==="
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

echo ""
echo "=== nvidia-smi ==="
nvidia-smi -L

echo ""
echo "=== torch CUDA check ==="
/home1/doyoonkim/miniconda3/envs/rac/bin/python -c "
import torch
print('torch.cuda.is_available():', torch.cuda.is_available())
print('torch.cuda.device_count():', torch.cuda.device_count())
import os
print('CUDA_VISIBLE_DEVICES in env:', os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET'))
if torch.cuda.is_available():
    print('device_capability:', torch.cuda.get_device_capability(0))
    print('device_name:', torch.cuda.get_device_name(0))
"

echo ""
echo "=== deepspeed import test ==="
/home1/doyoonkim/miniconda3/envs/rac/bin/python -c "
import os
print('CUDA_VISIBLE_DEVICES before import:', os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET'))
import deepspeed
print('deepspeed imported OK')
print('HAS_TRITON:', deepspeed.HAS_TRITON)
"

echo "##### END #####"
