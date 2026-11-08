#!/bin/bash
#SBATCH --job-name=rb2048
#SBATCH --partition=natStr
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/t512a04/users/doyoon/logs/%x_%j.out
# Qwen3-1.7B rebuttal set, rollout 2048, sidecar (fixed sync), latest code, one run per GPU:
#   s80: {cubic,linear} x reach step E {128,256,512,1024}, schedule applied every 8 steps (PGD off)
#   scout: s70 SCOUT calib batch 2/8/16, same-code SCOUT references s70 / s80 (calib 4),
#          s70 ablations: freeze after target, OPD loss off, off-policy (training doc) screening
# usage: sbatch [--nodes=N] rb2048.sbatch <s80|scout|all>   (4 runs per node)
#        rb2048_extra.sh <jobid> <node0>,<node1> s80        (same runs on idle nodes of another job)
# Resubmit to continue: every run resumes from its newest 256-step checkpoint.
U=/scratch/t512a04/users/doyoon
export RB_RUNS=$(bash $U/jobs/rb2048_runs.sh ${1:?s80|scout|all})
# --cpus-per-task: without it each node task is pinned to a single core (CPU 0) for everything on the node
srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE bash $U/jobs/rb2048_node.sh
echo "=== RB2048 EXIT: $? ==="
