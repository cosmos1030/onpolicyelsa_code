#!/bin/bash
#SBATCH --job-name=migrate_cache
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/migrate_cache_%j.out
exec 2>&1
export TOKENIZERS_PARALLELISM=false
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
/home1/doyoonkim/miniconda3/envs/rac/bin/python /home1/doyoonkim/projects/elsa/scripts/migrate_cache_filter_short.py
echo "##### END #####"
