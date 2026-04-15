#!/bin/bash
#SBATCH --job-name=mnist_gpu_test
#SBATCH --partition=A6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs/%j_mnist_gpu_test.out
#SBATCH -t 0-00:20:00

set -euo pipefail

mkdir -p /home1/doyoonkim/projects/RAC/open-r1-main/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/RAC/open-r1-main

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MNIST_GPU_TRAIN_SAMPLES=4096
export MNIST_GPU_EVAL_SAMPLES=1024
export MNIST_GPU_BATCH_SIZE=128
export MNIST_GPU_EPOCHS=2

echo "=== MNIST GPU smoke test ==="
echo "Node: $(hostname)"
date

python scripts/debugging/test_mnist_gpu.py

echo "=== Done ==="
date
