#!/bin/bash
#SBATCH --job-name=check_dataset
#SBATCH --partition=cpu-max16
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:15:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/check_dataset_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects
python elsa/scripts/check_dataset_stats_fast.py
