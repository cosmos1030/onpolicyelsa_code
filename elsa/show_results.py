#!/usr/bin/env python3
"""Show MATH-500 eval results for all ELSA models."""
import json
import glob
import os

EVAL_DIR = "/home1/doyoonkim/projects/elsa/eval_results"
MODEL_DIR = "/home1/doyoonkim/projects/elsa/models"

def find_score(eval_dir):
    from pathlib import Path
    files = sorted(Path(eval_dir).rglob("results_*.json"))
    if not files:
        return None
    data = json.load(open(files[-1]))
    results = data.get("results", {})
    key = "lighteval|math_500|0"
    if key in results:
        return results[key].get("pass@k:k=1&n=1")
    return None

rows = []

# Saved ELSA models
saved_models = sorted(os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else []
for model_tag in saved_models:
    eval_path = os.path.join(EVAL_DIR, model_tag)
    score = find_score(eval_path) if os.path.isdir(eval_path) else None
    score_str = f"{score*100:.1f}%" if score is not None else "—"
    rows.append((model_tag, score_str))

# Dense baseline
dense_tag = "DeepSeek-R1-Distill-Qwen-1.5B_dense"
dense_eval = os.path.join(EVAL_DIR, dense_tag)
dense_score = find_score(dense_eval) if os.path.isdir(dense_eval) else None
dense_str = f"{dense_score*100:.1f}%" if dense_score is not None else "—"

print("\n" + "="*80)
print(f"{'Model':<65} {'MATH-500':>10}")
print("="*80)
print(f"{'[DENSE] DeepSeek-R1-Distill-Qwen-1.5B':<65} {dense_str:>10}")
print("-"*80)
for tag, score_str in rows:
    short = tag.replace("DeepSeek-R1-Distill-Qwen-1.5B_pruned0.5_", "")
    print(f"{short:<65} {score_str:>10}")
print("="*80 + "\n")
