#!/bin/bash
# Source this at the end of SLURM jobs to register results in rundb.
# Usage: source rundb_post_job.sh <model> <sparsity> <badge> <name> <sub> <wbid> [project]
#
# Example:
#   source rundb_post_job.sh qwen3_4b 0.5 tropkd_mag "TR-GMP+OPKD (Mag, Global) kl=0.01" \
#       "magnitude · global · kl=0.01" "$WANDB_RUN_ID"
#
# This script:
#   1. Adds the run to results_db.json as pending
#   2. Immediately syncs metrics from wandb
#   3. Rebuilds artifact_built.html
#   (Artifact redeploy is handled by the cloud routine)

_RUNDB_MODEL=${1}
_RUNDB_SPARSITY=${2}
_RUNDB_BADGE=${3}
_RUNDB_NAME=${4}
_RUNDB_SUB=${5}
_RUNDB_WBID=${6}
_RUNDB_PROJECT=${7:-"reasoning_qwen3_4b"}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
SCRIPTS=/home1/doyoonkim/projects/elsa/scripts

echo "=== rundb: registering result ==="
echo "  model=$_RUNDB_MODEL tier=S$(python3 -c "print(int($_RUNDB_SPARSITY*100))")"
echo "  badge=$_RUNDB_BADGE  name=$_RUNDB_NAME  wbid=$_RUNDB_WBID"

cd "$SCRIPTS"
$PYTHON rundb/cli.py add-run \
    --model "$_RUNDB_MODEL" \
    --sparsity "$_RUNDB_SPARSITY" \
    --badge "$_RUNDB_BADGE" \
    --name "$_RUNDB_NAME" \
    --sub "$_RUNDB_SUB" \
    --wbid "$_RUNDB_WBID" \
    --project "$_RUNDB_PROJECT" \
    --sync \
    2>&1 && echo "rundb: OK" || echo "rundb: WARN — add-run failed (non-fatal)"
