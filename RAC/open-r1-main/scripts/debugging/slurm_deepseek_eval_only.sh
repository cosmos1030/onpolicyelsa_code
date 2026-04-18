#!/bin/bash
#SBATCH --job-name=deepseek_eval
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs/%j_deepseek_eval_only.out
#SBATCH -t 0-02:00:00

# Usage: sbatch slurm_deepseek_eval_only.sh <sparsity>
# e.g.:  sbatch slurm_deepseek_eval_only.sh 30

set -euo pipefail
SPARSITY=${1:-30}
echo "Node: $(hostname), Sparsity: ${SPARSITY}%"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/RAC/open-r1-main

MODEL_PATH="models/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_pruned_${SPARSITY}_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__OpenR1-Math-220k"

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<EOF
import sys, os
sys.path.insert(0, "/home1/doyoonkim/projects/RAC/open-r1-main/src/open_r1/utils")
os.chdir("/home1/doyoonkim/projects/RAC/open-r1-main")

from lighteval_math500 import run_lighteval_math500

model_path = "$MODEL_PATH"
print(f"Evaluating: {model_path}")

pass_at_1 = run_lighteval_math500(
    model_path=model_path,
    output_dir=f"{model_path}/lighteval_math500",
    max_new_tokens=30000,
    log_to_wandb=False,
)
print(f"MATH-500 pass@1 (sparsity=${SPARSITY}%): {pass_at_1}")
EOF

echo "=== Done ==="
date
