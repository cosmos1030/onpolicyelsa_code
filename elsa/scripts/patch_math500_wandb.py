"""Re-run math500 with max_model_length=32768 and patch into existing wandb run."""
import argparse, math, os, sys
sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
from lib.lighteval_5bench import _run_lighteval, _parse_results
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--wandb_run_id', required=True)
parser.add_argument('--out_dir', required=True)
parser.add_argument('--gpu_util', type=float, default=0.85)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
print(f"Running math500: max_new_tokens=32768, max_model_length=32768, gpu_util={args.gpu_util}", flush=True)

rc = _run_lighteval(
    args.model_path,
    "lighteval|math_500|0|0",
    args.out_dir,
    max_new_tokens=32768,
    gpu_util=args.gpu_util,
    max_samples=None,
    max_model_length=32768,
)
if rc != 0:
    print(f"[ERROR] lighteval exited with {rc}")
    sys.exit(1)

r = _parse_results(args.out_dir)
v = r.get("lighteval|math_500|0", {}).get("pass@k:k=1&n=1")
math500 = float(v) if v is not None else float("nan")
print(f"math500 = {math500:.4f}", flush=True)

wandb.init(
    id=args.wandb_run_id,
    resume='must',
    project='reasoning_pruning_v1',
    entity='dyk6208-gwangju-institute-of-science-and-technology',
)
wandb.log({'lighteval/math500': math500, 'lighteval/math500_32k_ctx': math500})
wandb.finish()
print("Done — wandb run updated.", flush=True)
