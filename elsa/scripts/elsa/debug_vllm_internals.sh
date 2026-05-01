#!/bin/bash
#SBATCH --job-name=vllm_internals
#SBATCH --partition=A5000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j_vllm_internals.out
#SBATCH -t 0-00:10:00

set -euo pipefail
echo "Node: $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<'EOF'
from vllm import LLM
import torch

model_path = "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.3,
          enforce_eager=True, trust_remote_code=True)

engine = llm.llm_engine
print(f"llm_engine type: {type(engine)}")
print(f"llm_engine attrs: {[a for a in dir(engine) if not a.startswith('__')]}")

# Try to find model executor path
for attr in ['model_executor', 'engine_core', 'executor', '_executor']:
    if hasattr(engine, attr):
        obj = getattr(engine, attr)
        print(f"\nFound: llm_engine.{attr} = {type(obj)}")
        print(f"  attrs: {[a for a in dir(obj) if not a.startswith('__')]}")
        for attr2 in ['driver_worker', '_driver_worker', 'workers']:
            if hasattr(obj, attr2):
                obj2 = getattr(obj, attr2)
                print(f"  Found: .{attr2} = {type(obj2)}")
                for attr3 in ['model_runner', 'model']:
                    if hasattr(obj2, attr3):
                        obj3 = getattr(obj2, attr3)
                        print(f"    Found: .{attr3} = {type(obj3)}")
EOF
echo "=== Done ==="
