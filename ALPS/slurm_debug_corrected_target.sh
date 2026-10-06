#!/bin/bash
#SBATCH --job-name=alps_ct_debug
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91,n61,n64
#SBATCH --output=/home1/doyoonkim/projects/ALPS/logs_debug/%j.out
exec 2>&1

mkdir -p /home1/doyoonkim/projects/ALPS/logs_debug

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd /home1/doyoonkim/projects/ALPS

echo "=== baseline ALPS, nsamples=4 (smoke test) ==="
$PYTHON qwen3_alps.py "$MODEL" 0.5 --data_path "$DATA" --nsamples 4 --rho 300.0 --seed 42

echo "=== corrected_target ALPS, nsamples=4 (smoke test) ==="
$PYTHON qwen3_alps.py "$MODEL" 0.5 --data_path "$DATA" --nsamples 4 --rho 300.0 --seed 42 --corrected_target

echo "##### END #####"
