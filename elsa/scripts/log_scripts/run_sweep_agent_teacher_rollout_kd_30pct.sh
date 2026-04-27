#!/bin/bash
#SBATCH --job-name=swp_tchr_kd30
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home/doyoonkim/onpolicyelsa_code/output_slurm/%j.out
#SBATCH -t 0-12:00:00

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh 2>/dev/null || true
cd /home/doyoonkim/onpolicyelsa_code/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/home/doyoonkim/.conda/envs/rac/lib/glibc_compat.so
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

/home/doyoonkim/.conda/envs/rac/bin/wandb agent \
    dyk6208-gwangju-institute-of-science-and-technology/elsa_qwen3_0.6b/1c8sdw92

echo "##### END #####"
date
