"""Run full MMLU-Redux (57 subsets) on dense model to verify MMLU_BATCH=10 OOM fix."""
import os
import sys
import math
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
from lib.lighteval_bench import MMLU_SUBSETS, _run_lighteval, _parse_results

import wandb

MODEL_PATH = (
    "/home1/doyoonkim/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/"
    "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
)
OUT_BASE = "/home1/doyoonkim/projects/elsa/eval_outputs/dense_1.5b_mmlu_full"

import torch
free, total = torch.cuda.mem_get_info(0)
gpu_util = round((free / total) * 0.92, 4)
logger.info(f"GPU: {free/1e9:.1f}GB free / {total/1e9:.1f}GB total → gpu_util={gpu_util}")

wandb.init(
    project="reasoning_pruning_v1",
    name="dense_1.5b_mmlu_full",
    entity="dyk6208-gwangju-institute-of-science-and-technology",
    config={"model": "DeepSeek-R1-1.5B", "sparsity": 0.0, "benchmark": "mmlu_full"},
)

MMLU_BATCH = 10
mmlu_batches = [MMLU_SUBSETS[i:i + MMLU_BATCH] for i in range(0, len(MMLU_SUBSETS), MMLU_BATCH)]
logger.info(f"MMLU: {len(MMLU_SUBSETS)} subsets → {len(mmlu_batches)} batches of up to {MMLU_BATCH}")

mmlu_scores = []
failed_batches = []

for batch_i, subset_batch in enumerate(mmlu_batches):
    batch_task = ",".join(f"lighteval|mmlu_redux_2:{s}|0|0" for s in subset_batch)
    batch_dir = os.path.join(OUT_BASE, f"mmlu_batch{batch_i}")
    os.makedirs(batch_dir, exist_ok=True)
    logger.info(f"  batch {batch_i+1}/{len(mmlu_batches)}: {subset_batch}")

    rc = _run_lighteval(MODEL_PATH, batch_task, batch_dir, 4096, gpu_util, None)
    if rc != 0:
        logger.error(f"  batch {batch_i} FAILED with exit code {rc}")
        failed_batches.append(batch_i)
        continue

    r = _parse_results(batch_dir)
    batch_scores = []
    for key, val in r.items():
        if "mmlu_redux_2" in key:
            score = val.get("acc_norm", val.get("acc"))
            if score is not None and not (isinstance(score, float) and math.isnan(score)):
                mmlu_scores.append(float(score))
                batch_scores.append(float(score))
    if batch_scores:
        logger.info(f"  batch {batch_i} avg: {sum(batch_scores)/len(batch_scores):.4f} ({len(batch_scores)} subsets)")

mmlu_avg = sum(mmlu_scores) / len(mmlu_scores) if mmlu_scores else float("nan")
logger.info(f"\nMMlu-Redux avg: {mmlu_avg:.4f} ({len(mmlu_scores)}/57 subsets)")
if failed_batches:
    logger.warning(f"Failed batches: {failed_batches}")

wandb.log({
    "lighteval/mmlu_redux": mmlu_avg,
    "mmlu_subsets_completed": len(mmlu_scores),
    "mmlu_batches_failed": len(failed_batches),
})
wandb.finish()
logger.info("Done.")
