#!/bin/bash
#SBATCH --job-name=vllm_internals2
#SBATCH --partition=A5000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j_vllm_internals2.out
#SBATCH -t 0-00:10:00

set -euo pipefail
echo "Node: $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<'EOF'
import os
os.environ["VLLM_USE_V1"] = "0"  # force legacy engine

from vllm import LLM
import torch

model_path = "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.3,
          enforce_eager=True, trust_remote_code=True)

engine = llm.llm_engine
print(f"llm_engine type: {type(engine)}")

# Try legacy path
for path in [
    "model_executor.driver_worker.model_runner.model",
    "model_executor._driver_worker.model_runner.model",
]:
    try:
        obj = engine
        for attr in path.split("."):
            obj = getattr(obj, attr)
        print(f"SUCCESS: llm_engine.{path} = {type(obj)}")
        # Try to list some params
        params = list(obj.named_parameters())
        print(f"  num params: {len(params)}, first: {params[0][0]}")
        break
    except AttributeError as e:
        print(f"FAIL: {path} -> {e}")
EOF
echo "=== Done ==="
