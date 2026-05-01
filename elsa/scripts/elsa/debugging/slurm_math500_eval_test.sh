#!/bin/bash
#SBATCH --job-name=elsa_math500_TEST
#SBATCH --partition=A100-40GB-PCIe
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j.out
#SBATCH -t 0-00:30:00

set -euo pipefail
mkdir -p /home1/doyoonkim/projects/elsa/output_qwen

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

# Test run_lighteval_math500 directly on a local model (no main.py involved)
LOCAL_MODEL=/home1/doyoonkim/projects/RAC/open-r1-main/models/Qwen3-0_pruned_30_all_tokens20000_prunemethod_SparseGPT_thirds_1_2_3__OpenR1-Math-220k

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<'PYEOF'
import sys
sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
from lib.lighteval_math500 import run_lighteval_math500

model_path = "/home1/doyoonkim/projects/RAC/open-r1-main/models/Qwen3-0_pruned_30_all_tokens20000_prunemethod_SparseGPT_thirds_1_2_3__OpenR1-Math-220k"

score = run_lighteval_math500(
    model_path=model_path,
    output_dir=model_path + "/lighteval_test",
    max_new_tokens=4096,
    log_to_wandb=False,
)
print(f"RESULT: pass@1 = {score}")
PYEOF

date
echo "=== Test done ==="