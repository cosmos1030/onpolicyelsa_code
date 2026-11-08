#!/bin/bash
#SBATCH --job-name=rb_maskquality_1.7b_s70_reach_vs_final
#SBATCH --partition=A6000,RTX6000ADA
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out
exec 2>&1
# Usage: sbatch slurm_mask_quality.sh <REACH_DIR> <FINAL_DIR> <TAG>
REACH=${1:?reach}; FINAL=${2:?final}; TAG=${3:?tag}
LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"; mkdir -p $LOCAL_JOB_BASE/slurm
NFS_LOG="/home1/doyoonkim/projects/elsa/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
trap 'cp "$LOCAL_JOB_BASE/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$NFS_LOG" 2>/dev/null || true' EXIT
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "NODE=$(hostname) JOB=$SLURM_JOB_ID reach=$REACH final=$FINAL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd /home1/doyoonkim/projects/elsa
mkdir -p rebuttal/results
/home1/doyoonkim/miniconda3/envs/rac/bin/python rebuttal/mask_quality.py \
  --dense /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --reach "$REACH" --final "$FINAL" --out rebuttal/results/maskquality_${TAG}_${SLURM_JOB_ID}.json
echo "=== EXIT $? ==="
