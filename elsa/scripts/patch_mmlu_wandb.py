"""Run MMLU-only and patch the result into an existing wandb run."""
import argparse, math, os, sys
sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
from lib.lighteval_bench import MMLU_SUBSETS, _run_lighteval, _parse_results
import torch, wandb

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--wandb_run_id', required=True)
parser.add_argument('--out_base', required=True)
parser.add_argument('--gpu_util', type=float, default=0.85)
args = parser.parse_args()

MMLU_BATCH = 5
mmlu_batches = [MMLU_SUBSETS[i:i+MMLU_BATCH] for i in range(0, len(MMLU_SUBSETS), MMLU_BATCH)]
print(f"MMLU: {len(MMLU_SUBSETS)} subsets → {len(mmlu_batches)} batches of {MMLU_BATCH}", flush=True)

mmlu_scores, failed = [], []
for batch_i, subset_batch in enumerate(mmlu_batches):
    batch_task = ','.join(f'lighteval|mmlu_redux_2:{s}|0|0' for s in subset_batch)
    batch_dir = os.path.join(args.out_base, f'mmlu_batch{batch_i}')
    os.makedirs(batch_dir, exist_ok=True)
    print(f"  batch {batch_i+1}/{len(mmlu_batches)}: {subset_batch}", flush=True)

    rc = _run_lighteval(args.model_path, batch_task, batch_dir, 4096, args.gpu_util, None)
    if rc != 0:
        print(f"  [ERROR] batch {batch_i} exited with {rc}", flush=True)
        failed.append(batch_i)
        continue

    r = _parse_results(batch_dir)
    for key, val in r.items():
        if 'mmlu_redux_2' in key:
            s = val.get('acc_norm', val.get('acc'))
            if s is not None and not (isinstance(s, float) and math.isnan(s)):
                mmlu_scores.append(float(s))

mmlu_avg = sum(mmlu_scores) / len(mmlu_scores) if mmlu_scores else float('nan')
print(f"\nMMlu-Redux avg: {mmlu_avg:.4f} ({len(mmlu_scores)}/57 subsets, {len(failed)} batches failed)", flush=True)

wandb.init(
    id=args.wandb_run_id,
    resume='must',
    project='reasoning_pruning_v1',
    entity='dyk6208-gwangju-institute-of-science-and-technology',
)
wandb.log({'lighteval/mmlu_redux': mmlu_avg, 'mmlu_subsets_completed': len(mmlu_scores)})
wandb.finish()
print("Done — wandb run updated.", flush=True)
