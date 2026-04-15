#!/bin/bash
#SBATCH --job-name=model_to_cuda
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs/%j_model_to_cuda.out
#SBATCH -t 0-00:15:00
#SBATCH --nodelist=n52

echo "Node: $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

unset PYTORCH_CUDA_ALLOC_CONF
unset SLURM_LOCALID SLURM_PROCID SLURM_STEP_ID SLURM_NODEID

/home1/doyoonkim/miniconda3/envs/rac/bin/python \
    /home1/doyoonkim/projects/RAC/open-r1-main/scripts/debugging/test_model_to_cuda.py

echo "=== Done ==="
date
