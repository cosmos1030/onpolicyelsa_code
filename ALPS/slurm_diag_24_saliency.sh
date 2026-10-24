#!/bin/bash
#SBATCH --job-name=diag24
#SBATCH --partition=A6000,L40S
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=n3,n42,n43,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n84,n90,n91
#SBATCH --output=/home1/doyoonkim/projects/ALPS/logs/diag24_%x_%j.out
exec 2>&1
# Usage: sbatch --job-name=diag24_<crit>_<mode> slurm_diag_24_saliency.sh <criterion> <zero|refit>
CRIT=${1:?criterion}; MODE=${2:-zero}
export HF_HOME=/home1/doyoonkim/.cache/huggingface HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /home1/doyoonkim/projects/ALPS
/home1/doyoonkim/miniconda3/envs/rac/bin/python -u diag_24_saliency.py \
    /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --criterion $CRIT --mode $MODE \
    --data_path /home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    --out /home1/doyoonkim/projects/ALPS/results/diag24/${CRIT}_${MODE}.json
