#!/bin/bash
#SBATCH --job-name=build_ourmask_init
#SBATCH --partition=A6000,L40S
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclude=n3,n42,n43,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n84,n90,n91
#SBATCH --output=/home1/doyoonkim/projects/ALPS/logs/build_ourmask_init_%j.out
exec 2>&1
# Our dense->2:4 run's final mask (job 1156578), weights re-solved from DENSE by masked least squares
# (ALPS-style sequential refit) -- init for the mask-vs-weight-path control.
export HF_HOME=/home1/doyoonkim/.cache/huggingface HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /home1/doyoonkim/projects/ALPS
/home1/doyoonkim/miniconda3/envs/rac/bin/python -u diag_24_saliency.py \
    /home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --criterion external --mask_from /home1/doyoonkim/projects/elsa/models/gmp_1.7b_n24_s50pct_lr5e-05_onpol_lmda0.33_20261016_085231_p1917729 --mode refit \
    --data_path /home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    --save /home1/doyoonkim/projects/elsa/models/qwen3_1.7b_n24_ourmask_refit_init --out /home1/doyoonkim/projects/ALPS/results/diag24/ourmask_refit.json
