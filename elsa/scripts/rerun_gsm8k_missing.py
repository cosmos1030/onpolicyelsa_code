"""One-off: rerun just the GSM8K lighteval task for a specific saved model and
log the result into the existing wandb run (resume), instead of a full
5-benchmark re-eval.

Usage: python rerun_gsm8k_missing.py <model_path> <wandb_project> <wandb_run_id> <out_base>
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.lighteval_bench import _run_lighteval, _parse_results

import wandb

model_path, wandb_project, wandb_run_id, out_base = sys.argv[1:5]

os.makedirs(out_base, exist_ok=True)
rc, elapsed = _run_lighteval(
    model_path, "lighteval|gsm8k|0|0", out_base,
    max_new_tokens=2048, gpu_util=0.85, max_model_length=4096,
)
if rc != 0:
    print(f"lighteval gsm8k exited with code {rc}, not logging")
    sys.exit(1)

r = _parse_results(out_base)
t = r.get("lighteval|gsm8k|0", {})
v = t.get("extractive_match", t.get("acc"))
print(f"gsm8k = {v}")

run = wandb.init(project=wandb_project, id=wandb_run_id, resume="allow",
                  entity="dyk6208-gwangju-institute-of-science-and-technology")
wandb.log({"lighteval/gsm8k": float(v), "eval_time_sec/gsm8k": elapsed})
wandb.finish()
print("wandb updated")
