#!/bin/bash
#SBATCH --job-name=mlp_gnpcg_test
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91,n61,n64
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/mlp_gnpcg_test_%j.out
exec 2>&1

# Quick sanity/correctness test for the MLP-joint Gauss-Newton PCG PoC
# (mlp_joint_gnpcg.py) -- single layer, small nsamples/pcg_iters, just to
# confirm it runs end-to-end and the recon error trajectory is sane
# (decreasing, not NaN/exploding) before committing to the full sweep.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
DENSE_MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_1.7b_alps_s50pct"
DATA="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"

export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/ALPS

$PYTHON mlp_joint_gnpcg.py "$DENSE_MODEL" "$ALPS_MODEL" \
    --data_path "$DATA" \
    --layer_idx 12 \
    --pcg_iters 1,2,5 \
    --gn_outer 1 \
    --damping 1e-3 \
    --nsamples 16 \
    --heldout 8 \
    --seqlen 2048

echo "##### END #####"
