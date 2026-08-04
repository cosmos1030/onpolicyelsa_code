#!/bin/bash
#SBATCH --job-name=fix_macko_env
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/fix_macko_env_%j.out
#SBATCH --exclude=n3,n42,n51,n52,n54,n55,n58,n60,n76,n77,n80
exec 2>&1

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader

MACKO_PY=/home1/doyoonkim/miniconda3/envs/macko/bin/python
MACKO_PIP=/home1/doyoonkim/miniconda3/envs/macko/bin/pip

echo "=== Current torch in macko env ==="
$MACKO_PY -c "import torch; print('torch:', torch.__version__)" 2>&1 || echo "torch import failed"

echo "=== Reinstalling torch 2.8.0+cu126 ==="
$MACKO_PIP install torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126 --no-deps -q

echo "=== Reinstalling macko_spmv ==="
cd /home1/doyoonkim/projects/macko_spmv
$MACKO_PIP install -e . -q

echo "=== Testing import ==="
$MACKO_PY -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'device:', torch.cuda.get_device_name(0))"
$MACKO_PY -c "import macko_spmv; print('macko_spmv OK')"

echo "##### END #####"
