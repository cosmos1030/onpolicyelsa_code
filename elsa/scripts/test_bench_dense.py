"""Quick smoke-test: run all 6 benchmarks (4 samples each) on dense model and log to wandb."""
import os, sys
sys.path.insert(0, "/home1/doyoonkim/projects/elsa")

MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
OUT_BASE = "/local-data/user-data/doyoonkim/test_bench_dense"
PROJECT = "reasoning_pruning_v1"
RUN_NAME = "dense_1.5b_bench_smoke"

import wandb
wandb.init(project=PROJECT, name=RUN_NAME, entity="dyk6208-gwangju-institute-of-science-and-technology",
           config={"model": "DeepSeek-R1-Distill-Qwen-1.5B", "sparsity": 0.0, "test": True})

from lib.lighteval_bench import run_lighteval_bench
metrics = run_lighteval_bench(
    model_path=MODEL_PATH,
    out_base=OUT_BASE,
    gpu_util=0.85,
    max_samples=4,
    log_to_wandb=True,
)

print("\n=== Results ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

wandb.finish()
