#!/bin/bash
#SBATCH --job-name=eval32b-4models
#SBATCH --partition=natStr
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/t512a04/users/doyoon/logs/%x_%j.out
# Long-profile avg4 evals (MATH500, IFEval, LCB, GSM8K) of all four 32B models in ONE job: one model per
# node, and on each node lcb on GPU0-1 next to math500 -> ifeval -> gsm8k on GPU2-3 (see eval_32b_long.sbatch).
# Optional 1st arg: samples per task (smoke test).
U=/scratch/t512a04/users/doyoon
export EVAL_MODELS="dense32b:$U/models/Qwen3-32B
scout32b:$U/models_out/gmp_32b_s70pct_lr0.0001_onpol_lmda0.33_20261108_011330_p2844593
alpssft32b:$U/models_out/gmp_32b_s70pct_lr0.0001_onpol_lmda0.33_20261108_013003_p378358
alps32b:$U/models_out/qwen3_32b_alps_s70pct"
export EVAL_SMOKE=${1:-}
srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE bash $U/jobs/eval_node.sh
echo "=== EVAL EXIT: $? ==="
