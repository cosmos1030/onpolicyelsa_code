#!/bin/bash
#SBATCH --job-name=ob_reeval
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --exclude=n52,n55,n58,n80
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs/ob_reeval_%j.out
exec 2>&1

# Re-evaluate OpenBookQA (acc_norm) for a single HF model.
# Usage: sbatch slurm_openbookqa_reeval.sh <HF_REPO> <WANDB_RUN_ID>
# e.g.:  sbatch slurm_openbookqa_reeval.sh cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260710_171528 qd1117ir

HF_REPO=${1:?"Usage: sbatch slurm_openbookqa_reeval.sh <HF_REPO> <WANDB_RUN_ID>"}
WBID=${2:?"Usage: sbatch slurm_openbookqa_reeval.sh <HF_REPO> <WANDB_RUN_ID>"}

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
OUT=/local-data/user-data/$USER/job_$SLURM_JOB_ID/ob_reeval

echo "=== OpenBookQA Re-Eval ==="
echo "HF_REPO=$HF_REPO  WBID=$WBID  NODE=$(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
date

$PYTHON -m lm_eval \
    --model hf \
    --model_args "pretrained=${HF_REPO},dtype=bfloat16" \
    --tasks openbookqa \
    --num_fewshot 0 \
    --batch_size 16 \
    --output_path "$OUT"

echo ""
echo "=== RESULT ==="
$PYTHON - <<EOF
import json, glob, os
files = glob.glob("${OUT}/**/*.json", recursive=True)
for f in files:
    d = json.load(open(f))
    results = d.get("results", {})
    ob = results.get("openbookqa", {})
    acc_norm = ob.get("acc_norm,none")
    acc      = ob.get("acc,none")
    print(f"WBID=${WBID}  acc_norm={acc_norm}  acc={acc}")
    print(f"UPDATE runs_db.json: ob = {round(acc_norm*100, 1) if acc_norm else 'N/A'}")
EOF

echo "=== Done ==="
date
