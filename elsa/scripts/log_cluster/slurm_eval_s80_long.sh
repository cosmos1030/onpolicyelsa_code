#!/bin/bash
#SBATCH --job-name=eval_s80_long
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/eval_s80_long_%x_%j.out
exec 2>&1

# SLURM wrapper for log_scripts/eval_s80_long.sh (written for a non-SLURM box).
# One arm per job so each can start on whichever single GPU frees first:
#   for a in base dpo_lr1e5_ep04 dpo_lr5e6_ep10; do
#     sbatch -J s80_$a elsa/scripts/log_cluster/slurm_eval_s80_long.sh $a; done
ARM=${1:?"usage: sbatch slurm_eval_s80_long.sh <base|dpo_lr1e5_ep04|dpo_lr5e6_ep10>"}

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

export REPO_ROOT=/home/doyoonkim/projects/onpolicyelsa_code
export PYTHON=$(which python)
export HF_HOME=/home/shared/huggingface
export OUT_ROOT=$HOME/elsa_eval_s80
# env.sh defaults this to 0 (a B200 workaround); vllm 0.10 in rac runs V1,
# matching slurm_eval_lighteval_only.sh.
export VLLM_USE_V1=1
export TMPDIR=/tmp/${USER}/job_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export WANDB_DIR=$TMPDIR

echo "host $(hostname)  job $SLURM_JOB_ID  gpus $CUDA_VISIBLE_DEVICES"
# CUDA_VISIBLE_DEVICES is already set by SLURM; don't pass a GPU id.
bash "$REPO_ROOT/log_scripts/eval_s80_long.sh" "$ARM"
CODE=$?
rm -rf "$TMPDIR"
exit $CODE
