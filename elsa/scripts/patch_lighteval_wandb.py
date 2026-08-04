"""Re-run all lighteval benchmarks and patch into existing wandb run (reasoning_pruning_v2)."""
import argparse, os, sys
sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
from lib.lighteval_bench import _run_lighteval, _parse_results, _compute_token_stats
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--wandb_run_id', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--gpu_util', type=float, default=0.85)
parser.add_argument('--project', default='reasoning_pruning_v2')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

benchmarks = [
    ("math500", "lighteval|math_500|0|0",           8192, 8192, ["pass@k:k=1&n=1"]),
    ("gpqa",    "lighteval|gpqa:diamond|0|0",       8192, 8192, ["gpqa_pass@k:k=1", "pass@k:k=1&n=1", "acc_norm", "acc"]),
    ("ifeval",  "lighteval|ifeval|0|0",             8192, 8192, ["prompt_level_strict_acc"]),
    ("lcb",     "lighteval|lcb:codegeneration|0|0", 8192, 8192, ["codegen_pass@1:16", "pass@1"]),
]

metrics = {}
for name, task_str, max_tok, ctx_len, correct_keys in benchmarks:
    out_dir = os.path.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== {name} ===", flush=True)
    rc, elapsed = _run_lighteval(args.model_path, task_str, out_dir, max_tok, args.gpu_util,
                                 max_samples=None, max_model_length=ctx_len)
    metrics[f"eval_time_sec/{name}"] = elapsed
    if rc != 0:
        print(f"[ERROR] {name} exited with {rc}", flush=True)
        continue
    r = _parse_results(out_dir)

    if name == "math500":
        v = r.get("lighteval|math_500|0", {}).get("pass@k:k=1&n=1")
        metrics["lighteval/math500"] = float(v) if v is not None else float("nan")
        metrics["math500_pass@1"] = metrics["lighteval/math500"]
    elif name == "gpqa":
        t = r.get("lighteval|gpqa:diamond|0", {})
        v = t.get("gpqa_pass@k:k=1", t.get("acc_norm", t.get("acc", t.get("pass@k:k=1&n=1"))))
        metrics["lighteval/gpqa_diamond"] = float(v) if v is not None else float("nan")
    elif name == "ifeval":
        t = r.get("lighteval|ifeval|0", {})
        v = t.get("prompt_level_strict_acc", t.get("acc"))
        metrics["lighteval/ifeval_prompt"] = float(v) if v is not None else float("nan")
    elif name == "lcb":
        t = r.get("lighteval|lcb:codegeneration|0", {})
        v = t.get("codegen_pass@1:16", t.get("pass@1"))
        metrics["lighteval/lcb"] = float(v) if v is not None else float("nan")

    metrics.update(_compute_token_stats(out_dir, name, max_tok, correct_keys))
    print(f"  result: {list(metrics.items())[-1]}", flush=True)

print(f"\nPatching wandb run {args.wandb_run_id} in project {args.project}...", flush=True)
wandb.init(
    id=args.wandb_run_id,
    resume='must',
    project=args.project,
    entity='dyk6208-gwangju-institute-of-science-and-technology',
)
wandb.log(metrics)
wandb.finish()
print("Done — wandb run patched.", flush=True)
for k, v in sorted(metrics.items()):
    if 'eval_time' not in k and 'token' not in k:
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
