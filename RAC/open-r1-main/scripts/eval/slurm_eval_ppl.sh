#!/bin/bash
#SBATCH --job-name=eval_ppl
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs/%j_eval_ppl.out

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

export PYTHONUNBUFFERED=1
export WANDB_API_KEY=$(cat ~/.wandb_api_key 2>/dev/null || echo "$WANDB_API_KEY")
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

cd /home1/doyoonkim/projects/RAC/open-r1-main

source /home1/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda activate rac

python scripts/eval_ppl_log_wandb.py
