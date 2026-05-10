#!/bin/bash
#SBATCH --job-name=flash_posids
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home1/doyoonkim/projects/elsa/debug_flash_posids_%j.out
#SBATCH -t 0-00:10:00

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa
export PYTHONUNBUFFERED=1
echo "Node: $(hostname)"
/home1/doyoonkim/miniconda3/envs/rac/bin/python -u debug_flash_posids.py
echo "##### END #####"
