#!/bin/bash
#SBATCH --job-name=alps32b-s70
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/t512a04/users/doyoon/logs/%x_%j.out
# One-shot ALPS s70 on Qwen3-32B (baseline for the 32B ALPS -> NTP/KD/OPD comparison). Same recipe as the
# paper's 8B ALPS (b200_scripts/alps_prune_qwen3_8b.sh): OT3/FineWeb 8:2 calibration, nsamples 128,
# seqlen 2048, rho 300, seed 42. Layer-wise, so one GPU; ALPS_GPU_EIGH=1 runs the float64 eigh on the GPU.
U=/scratch/t512a04/users/doyoon
E=/tmp/scout_env
fusermount -uz $E 2>/dev/null  # stale mount left by an earlier job on this node
mkdir -p $E && squashfuse $U/scout_env.sqfs $E || { echo "squashfuse mount failed"; exit 1; }
trap "fusermount -uz $E" EXIT
source $U/jobs/env_common.sh
export CUDA_VISIBLE_DEVICES=0 ALPS_GPU_EIGH=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd $U/code/ALPS
python -u qwen3_alps.py $U/models/Qwen3-32B 0.7 \
    --data_path $U/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    --nsamples 128 --rho 300.0 --seed 42 \
    --save $U/models_out/qwen3_32b_alps_s70pct \
    --wandb_project reasoning_qwen3_32b --run_name alps_32b_s70pct_hangang
echo "=== ALPS EXIT: $? ==="
