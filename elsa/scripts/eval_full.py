"""Full evaluation: PPL (c4/wikitext2) + zero-shot (9 tasks) + lighteval bench (5 tasks).

Usage:
    python eval_full.py \
        --model_path <path> \
        --wandb_project reasoning_pruning_v2 \
        --run_name dense \
        [--sparsity 0.0] \
        [--method dense] \
        [--gpu_util 0.85]
"""
import argparse
import gc
import json
import logging
import os
import sys
import types

import torch
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.eval import eval_ppl, eval_zero_shot
from lib.lighteval_bench import run_lighteval_bench

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ZERO_SHOT_TASKS = [
    "boolq", "rte", "hellaswag", "winogrande",
    "arc_challenge", "arc_easy", "openbookqa", "piqa", "race",
]

# Metric key to extract per task (lm_eval result dict format)
ZERO_SHOT_METRIC = {
    "boolq":         ("acc,none", "acc_norm,none"),
    "rte":           ("acc,none", "acc_norm,none"),
    "hellaswag":     ("acc_norm,none", "acc,none"),
    "winogrande":    ("acc,none", "acc_norm,none"),
    "arc_challenge": ("acc_norm,none", "acc,none"),
    "arc_easy":      ("acc_norm,none", "acc,none"),
    "openbookqa":    ("acc_norm,none", "acc,none"),
    "piqa":          ("acc_norm,none", "acc,none"),
    "race":          ("acc,none", "acc_norm,none"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--wandb_project", required=True)
    p.add_argument("--wandb_entity", default="dyk6208-gwangju-institute-of-science-and-technology")
    p.add_argument("--run_name", default=None, help="e.g. dense, sparsegpt_s50 (optional if --wandb_run_id given)")
    p.add_argument("--wandb_run_id", default=None, help="resume existing wandb run and log PPL there")
    p.add_argument("--method", default="dense", help="dense / sparsegpt / gmp / elsa")
    p.add_argument("--sparsity", type=float, default=0.0)
    p.add_argument("--gpu_util", type=float, default=0.85)
    p.add_argument("--tp_size", type=int, default=1, help="tensor parallel size for vLLM (lighteval)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds for a multi-seed lighteval variance run (e.g. 42,0,1) -- "
                        "overrides --seed for the lighteval section only (PPL/zero-shot still use --seed once). "
                        "Runs each seed back to back, logs per-seed metrics as '<metric>_seed<N>', and logs "
                        "'<metric>_mean'/'<metric>_std' across seeds to the same wandb run. Recommended for "
                        "AIME24/25 and GPQA (small sample counts, high per-problem variance); math500/ifeval/"
                        "gsm8k/lcb are fine with a single --seed given their sample counts.")
    p.add_argument("--skip_ppl", action="store_true")
    p.add_argument("--skip_zeroshot", action="store_true")
    p.add_argument("--skip_lighteval", action="store_true")
    p.add_argument("--max_samples", type=int, default=None, help="limit samples per lighteval benchmark (smoke test)")
    p.add_argument("--benchmarks", default=None, help="comma-separated subset of lighteval benchmark names to run (e.g. aime24,aime25); default runs all")
    p.add_argument("--out_base", default=None, help="base dir for lighteval outputs")
    p.add_argument("--flops", type=float, default=None,
                   help="Precomputed compute cost (FLOPs) for the pruning/calibration step that produced this "
                        "model, passed through by the caller (e.g. 2*n_params*n_tokens for one-shot methods "
                        "like ALPS/SparseGPT/Wanda/SparseLLM, 6*n_params*n_tokens for gradient fine-tuning). "
                        "Logged as-is; not computed here since this script doesn't know nsamples/seqlen.")
    p.add_argument("--hub_model_id", default=None, help="HF Hub repo id the caller already pushed the model to (logged into this run's summary).")
    p.add_argument("--hub_url", default=None, help="HF Hub URL the caller already pushed the model to (logged into this run's summary).")
    return p.parse_args()


def load_model(model_path, device="cuda", multi_gpu=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map="auto" if multi_gpu else device,
    )
    model.eval()
    # eval_ppl expects model.seqlen
    if not hasattr(model, "seqlen"):
        model.seqlen = 2048
    return model, tokenizer


def run_ppl(model, tokenizer, args_ns):
    logger.info("=== PPL eval (c4, wikitext2) ===")
    ppls = eval_ppl(args_ns, model, tokenizer, data_path=getattr(args_ns, "data_path", None))
    metrics = {
        "ppl/c4":        float(ppls.get("c4", float("nan"))),
        "ppl/wikitext2": float(ppls.get("wikitext2", float("nan"))),
    }
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.2f}")
    return metrics


def run_zeroshot(model, tokenizer, args_ns):
    logger.info("=== Zero-shot eval ===")
    results = eval_zero_shot(
        args_ns, args_ns.model_path, model, tokenizer,
        task_list=ZERO_SHOT_TASKS, num_fewshot=0,
    )
    metrics = {}
    for task in ZERO_SHOT_TASKS:
        task_result = results.get(task, {})
        primary_key, fallback_key = ZERO_SHOT_METRIC[task]
        v = task_result.get(primary_key, task_result.get(fallback_key))
        score = float(v) if v is not None else float("nan")
        short = task.replace("_", "").replace("arc", "arc_")
        metrics[f"zeroshot/{task}"] = score
        logger.info(f"  {task}: {score:.4f}")
    avg = [v for v in metrics.values() if v == v]  # filter nan
    metrics["zeroshot/avg"] = sum(avg) / len(avg) if avg else float("nan")
    logger.info(f"  avg: {metrics['zeroshot/avg']:.4f}")
    return metrics


def main():
    args = parse_args()

    run_label = args.run_name or args.wandb_run_id or f"{args.method}_s{args.sparsity}"
    out_base = args.out_base or os.path.join(
        "/local-data/user-data", os.environ.get("USER", "user"),
        f"eval_{run_label}",
    )
    os.makedirs(out_base, exist_ok=True)

    try:
        init_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            settings=wandb.Settings(init_timeout=300),
        )
        if args.wandb_run_id:
            # Resume existing training run to log PPL into same run
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"
            logger.info(f"Resuming wandb run {args.wandb_run_id}")
        else:
            init_kwargs["name"] = args.run_name or run_label
            init_kwargs["config"] = {
                "model_path": args.model_path,
                "method": args.method,
                "sparsity": args.sparsity,
            }
        run = wandb.init(**init_kwargs)
        use_wandb = True
    except Exception as e:
        logger.warning(f"wandb.init failed ({e}), continuing without wandb — results saved to JSON only")
        run = None
        use_wandb = False

    all_metrics = {
        "method":   args.method,
        "sparsity": args.sparsity,
    }
    if args.flops is not None:
        all_metrics["flops"] = args.flops
        if use_wandb:
            wandb.log({"flops": args.flops})
    if args.hub_model_id:
        all_metrics["hub_model_id"] = args.hub_model_id
        all_metrics["hub_model_url"] = args.hub_url
        if use_wandb:
            wandb.run.summary["hub_model_id"] = args.hub_model_id
            wandb.run.summary["hub_model_url"] = args.hub_url

    # ── PPL + zero-shot: load model via HF ──────────────────────────────────
    if not args.skip_ppl or not args.skip_zeroshot:
        # Use local c4 shard — compute nodes have no outbound internet access
        args_ns = types.SimpleNamespace(
            seed=args.seed,
            model_path=args.model_path,
            data_path="/home1/doyoonkim/projects/elsa/data/c4",
        )
        model, tokenizer = load_model(args.model_path, multi_gpu=args.tp_size > 1)

        if not args.skip_ppl:
            m = run_ppl(model, tokenizer, args_ns)
            all_metrics.update(m)
            if use_wandb:
                wandb.log(m)

        if not args.skip_zeroshot:
            m = run_zeroshot(model, tokenizer, args_ns)
            all_metrics.update(m)
            if use_wandb:
                wandb.log(m)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU memory freed.")

    # ── lighteval benchmarks: vLLM subprocess ───────────────────────────────
    # 7 tasks: math500, aime24, aime25, gpqa, ifeval, lcb, gsm8k
    if not args.skip_lighteval:
        only_tasks = args.benchmarks.split(",") if args.benchmarks else None
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
        logger.info(f"=== lighteval bench (seeds={seeds}) ===")

        per_seed_runs = []
        for s in seeds:
            lighteval_out = os.path.join(out_base, "lighteval") if len(seeds) == 1 \
                else os.path.join(out_base, "lighteval", f"seed{s}")
            m = run_lighteval_bench(
                model_path=args.model_path,
                out_base=lighteval_out,
                gpu_util=args.gpu_util,
                log_to_wandb=False,
                tp_size=args.tp_size,
                max_samples=args.max_samples,
                only_tasks=only_tasks,
                seed=s,
            )
            per_seed_runs.append(m)
            if len(seeds) == 1:
                all_metrics.update(m)
                if use_wandb:
                    wandb.log(m)
            else:
                seeded = {f"{k}_seed{s}": v for k, v in m.items()}
                all_metrics.update(seeded)
                if use_wandb:
                    wandb.log(seeded)

        if len(seeds) > 1:
            import math as _math
            all_keys = sorted({k for run in per_seed_runs for k in run})
            agg = {}
            for k in all_keys:
                vals = [run[k] for run in per_seed_runs if k in run and isinstance(run[k], (int, float))
                        and not (isinstance(run[k], float) and _math.isnan(run[k]))]
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
                agg[f"{k}_mean"] = mean
                agg[f"{k}_std"] = std
            all_metrics.update(agg)
            if use_wandb:
                wandb.log(agg)
            logger.info(f"[eval_full] multi-seed aggregate: {agg}")

    # Save summary JSON
    summary_path = os.path.join(out_base, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")
    if use_wandb:
        wandb.save(summary_path)
        wandb.finish()
    logger.info("=== eval_full.py done ===")


if __name__ == "__main__":
    main()
