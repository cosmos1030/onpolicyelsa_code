"""Long-profile lighteval for one model on a hangang node (eval_32b_long.sbatch runs two of these side by side).

  run:   python eval_long.py run   --model_dir M --out_dir O --tasks lcb --tp 2 --tag a
         -> O/<task>/ (lighteval output) and O/summary_<tag>.json
  merge: python eval_long.py merge --out_dir O --run_name eval_long_dense32b
         -> O/summary.json with avg4 (MATH500, IFEval, LCB, GSM8K) and avg5 (+GPQA), logged to wandb
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("U", "/scratch/t512a04/users/doyoon"), "code", "elsa"))

AVG4 = ["lighteval/math500", "lighteval/ifeval_prompt", "lighteval/lcb", "lighteval/gsm8k"]
AVG5 = AVG4 + ["lighteval/gpqa_diamond"]


def run(a):
    from lib.lighteval_bench import run_lighteval_bench
    os.makedirs(a.out_dir, exist_ok=True)
    m = run_lighteval_bench(model_path=a.model_dir, out_base=a.out_dir, gpu_util=a.gpu_util,
                            tp_size=a.tp, log_to_wandb=False, only_tasks=a.tasks.split(","),
                            seed=42, profile="long", max_samples=a.max_samples)
    with open(os.path.join(a.out_dir, f"summary_{a.tag}.json"), "w") as f:
        json.dump(m, f, indent=2)


def merge(a):
    m = {}
    for p in sorted(glob.glob(os.path.join(a.out_dir, "summary_*.json"))):
        with open(p) as f:
            m.update(json.load(f))
    for name, keys in (("avg4", AVG4), ("avg5", AVG5)):
        have = [m[k] for k in keys if k in m and m[k] == m[k]]
        if len(have) == len(keys):
            m[f"lighteval/{name}"] = sum(have) / len(have) * 100
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(m, f, indent=2)
    for k in AVG5 + ["lighteval/avg4", "lighteval/avg5"]:
        print(f"{k}: {m.get(k)}")
    if a.run_name:
        import wandb
        wandb.init(project="reasoning_qwen3_32b", name=a.run_name,
                   config={"eval_profile": "long", "model_dir": a.model_dir})
        wandb.log(m)
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "merge"])
    p.add_argument("--model_dir", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tasks", default="math500,gpqa,ifeval,lcb,gsm8k")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--gpu_util", type=float, default=0.9)
    p.add_argument("--tag", default="all")
    p.add_argument("--run_name", default=None)
    p.add_argument("--max_samples", type=int, default=None, help="smoke test only")
    a = p.parse_args()
    run(a) if a.mode == "run" else merge(a)
