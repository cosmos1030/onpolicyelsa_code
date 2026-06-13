"""
Evaluate dense (unpruned) model on MATH-500 across multiple generation seeds
to measure evaluation variance.

Usage:
    python eval_dense_variance.py [--seeds 0 1 2 3 42] [--max_samples 500]
"""
import argparse
import json
import tempfile
import numpy as np
import wandb

MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 42])
parser.add_argument("--max_samples", type=int, default=500)
parser.add_argument("--max_new_tokens", type=int, default=8192)
parser.add_argument("--temperature", type=float, default=0.6)
parser.add_argument("--wandb_project", type=str, default="gmp_qwen3_1.5b_v2")
args = parser.parse_args()

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.lighteval_math500 import run_lighteval_math500

run = wandb.init(
    project=args.wandb_project,
    entity="dyk6208-gwangju-institute-of-science-and-technology",
    name=f"dense_variance_n{args.max_samples}",
    config=vars(args),
)

scores = []
print(f"\nEvaluating dense model on MATH-500 ({args.max_samples} samples) × {len(args.seeds)} seeds")
print(f"Seeds: {args.seeds}\n")

for seed in args.seeds:
    with tempfile.TemporaryDirectory() as tmpdir:
        score = run_lighteval_math500(
            MODEL_PATH,
            output_dir=tmpdir,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            temperature=args.temperature,
            seed=seed,
            log_to_wandb=False,
        )
    scores.append(score)
    print(f"  seed={seed:>3}  pass@1={score:.4f}")
    wandb.log({"seed": seed, "pass@1": score})

mean = np.mean(scores)
std  = np.std(scores)
print(f"\nResult: mean={mean:.4f}  std={std:.4f}  min={min(scores):.4f}  max={max(scores):.4f}")
wandb.log({
    "dense/mean": mean,
    "dense/std":  std,
    "dense/min":  min(scores),
    "dense/max":  max(scores),
    "dense/scores": scores,
})

wandb.finish()
