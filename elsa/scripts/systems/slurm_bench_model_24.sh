#!/bin/bash
#SBATCH --job-name=e2e24
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1
OUTDIR=/home1/doyoonkim/projects/elsa/logs/systems; mkdir -p "$OUTDIR"
LB="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"; mkdir -p "$LB/slurm"
trap 'cp "$LB/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$OUTDIR/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" 2>/dev/null || true' EXIT
export TMPDIR=/tmp TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "NODE=$(hostname) JOB=$SLURM_JOB_ID"; nvidia-smi --query-gpu=name --format=csv,noheader
cd /home1/doyoonkim/projects/elsa
/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/systems/bench_model_24.py \
    --model /home1/doyoonkim/projects/elsa/models/scout_4b_2to4_best --out /home1/doyoonkim/projects/elsa/logs/systems/e2e24_${SLURM_JOB_ID}.json
echo "=== EXIT: $? ==="; echo "##### END #####"
