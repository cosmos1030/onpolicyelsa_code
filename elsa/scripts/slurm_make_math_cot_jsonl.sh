#!/bin/bash
#SBATCH --job-name=make_cot_jsonl
#SBATCH --partition=cpu-max16
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=output/%j.out
#SBATCH -t 0-02:00:00

mkdir -p output

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/elsa

/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/make_math_cot_jsonl.py \
    --output /home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl

echo "Done."