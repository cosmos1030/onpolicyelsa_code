#!/bin/bash
#SBATCH --job-name=gpu_mem_test
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs/%j_gpu_mem_test.out
#SBATCH -t 0-00:10:00
#SBATCH --exclude=n56,n58,n80

set -euo pipefail
echo "Node: $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

unset PYTORCH_CUDA_ALLOC_CONF
unset SLURM_LOCALID SLURM_PROCID SLURM_STEP_ID SLURM_NODEID

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<'EOF'
import os, torch

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'not set')}")
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'not set')}")

torch.cuda.init()
free, total = torch.cuda.mem_get_info(0)
print(f"GPU 0: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total")

allocated = []
for size_mb in [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]:
    try:
        t = torch.zeros(size_mb * 1024 * 256, dtype=torch.float32, device="cuda:0")  # size_mb MiB
        allocated.append(t)
        used = torch.cuda.memory_allocated(0) / 1024**2
        print(f"  Alloc {size_mb} MiB cumulative: OK  (torch.memory_allocated={used:.0f} MiB)")
    except Exception as e:
        print(f"  Alloc {size_mb} MiB: FAILED -> {e}")
        break

# Cleanup
del allocated
torch.cuda.empty_cache()

# Try single-shot large allocation
for size_mb in [500, 1000, 2000, 4000, 8000]:
    torch.cuda.empty_cache()
    try:
        t = torch.zeros(size_mb * 1024 * 256, dtype=torch.float32, device="cuda:0")
        print(f"Single-shot {size_mb} MiB: OK")
        del t
    except Exception as e:
        print(f"Single-shot {size_mb} MiB: FAILED -> {e}")
        break

print("Done")
EOF
echo "=== Test done ==="
date
