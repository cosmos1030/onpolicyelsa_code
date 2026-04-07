#!/bin/bash
#SBATCH --job-name=math500_dense_qwen3
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j.out
#SBATCH -t 0-01:00:00

set -euo pipefail
mkdir -p /home1/doyoonkim/projects/elsa/output_qwen

source ~/miniconda3/etc/profile.d/conda.sh

DENSE_MODEL=/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca

START=$(date +%s)
echo "=== Start: $(date) ==="

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<'PYEOF'
import sys, time
sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
from lib.lighteval_math500 import run_lighteval_math500

model_path = "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

t0 = time.time()
score = run_lighteval_math500(
    model_path=model_path,
    output_dir=model_path + "/lighteval_math500_32k",
    log_to_wandb=False,
)
elapsed = time.time() - t0
print(f"RESULT: pass@1 = {score}")
print(f"ELAPSED: {elapsed:.1f}s ({elapsed/60:.1f} min)")
PYEOF

END=$(date +%s)
echo "=== End: $(date) === Total: $(( (END - START) / 60 )) min ==="