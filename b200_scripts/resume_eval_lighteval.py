"""Resume the post-training eval that main.py runs after pruning, standalone.

main.py's post-pruning tail is: ppl -> zero-shot -> lighteval bench (5 quick
tasks) -> push_to_hub. If the container dies partway (as it did for
opkdfix_8b_s70_delta0.03 at 2026-09-06 11:33, mid-math500), the pruned model is
already saved on disk and everything after it is re-runnable without retraining.
This script re-runs the lighteval bench + hub upload against a saved model dir
and logs into the SAME wandb run, so the row lands next to its training curves.

lighteval caches generations under <model_dir>/<model_hash>/<task>/<sample_hash>,
keyed on the model args string -- so passing the same --gpu_util the crashed run
used makes an already-generated task (math500 here) skip generation and only
re-score. That's why gpu_util is an explicit arg instead of being recomputed
from free VRAM the way main.py does it.

Usage:
  python resume_eval_lighteval.py --model_dir <path> --wandb_run_id <id> \
      [--wandb_project ...] [--tasks math500,gpqa,ifeval,lcb,gsm8k] \
      [--tp_size 4] [--gpu_util 0.9368] [--seed 42] [--profile quick] \
      [--hub_repo cosmos1030/... | --no_hub]
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, "/NHNHOME/log-postech/doyoonkim/onpolicyelsa_code/elsa")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# CSV avg5 = mean of these five, in this order.
AVG5_KEYS = [
    "lighteval/math500",
    "lighteval/lcb",
    "lighteval/gpqa_diamond",
    "lighteval/gsm8k",
    "lighteval/ifeval_prompt",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="saved pruned model dir (has config.json)")
    p.add_argument("--wandb_run_id", default=None, help="resume this run; omit to skip wandb")
    p.add_argument("--wandb_project", default="reasoning_qwen3_8b_nostrip8192")
    p.add_argument("--tasks", default=None,
                   help="comma-separated subset of math500,gpqa,ifeval,lcb,gsm8k (default: all in profile)")
    p.add_argument("--tp_size", type=int, default=4)
    p.add_argument("--gpu_util", type=float, default=0.9368,
                   help="must match the crashed run's value to reuse its generation cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--profile", default="quick", choices=["quick", "official", "full"])
    p.add_argument("--hub_repo", default=None, help="HF repo id; auto-generated if omitted")
    p.add_argument("--no_hub", action="store_true", help="skip the HF upload")
    p.add_argument("--sparsity", type=float, default=0.7, help="only used for the auto repo name / commit msg")
    p.add_argument("--lr", type=float, default=1e-4, help="only used for the auto repo name / commit msg")
    p.add_argument("--kd_lambda", type=float, default=0.33, help="only used for the auto repo name")
    return p.parse_args()


def _fmt_float(v):
    # same formatting main.py uses for hub repo names: 0.33 -> 3e-1, 1e-4 -> 1e-4
    return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")


def main():
    args = parse_args()
    if not os.path.isfile(os.path.join(args.model_dir, "config.json")):
        sys.exit(f"no config.json in {args.model_dir} -- not a saved model dir")

    use_wandb = False
    if args.wandb_run_id:
        try:
            import wandb
            wandb.init(project=args.wandb_project, id=args.wandb_run_id, resume="allow",
                       settings=wandb.Settings(init_timeout=300))
            use_wandb = True
            logger.info(f"resumed wandb run {args.wandb_run_id} in {args.wandb_project}")
        except Exception as e:
            logger.warning(f"wandb resume failed ({e}); continuing without wandb")

    from lib.lighteval_bench import run_lighteval_bench

    only_tasks = args.tasks.split(",") if args.tasks else None
    out_base = os.path.join(args.model_dir, "lighteval_bench")
    logger.info(f"=== lighteval bench (profile={args.profile}, tasks={only_tasks or 'all'}, "
                f"tp={args.tp_size}, gpu_util={args.gpu_util:.4f}) ===")
    metrics = run_lighteval_bench(
        model_path=args.model_dir,
        out_base=out_base,
        gpu_util=args.gpu_util,
        tp_size=args.tp_size,
        log_to_wandb=use_wandb,
        only_tasks=only_tasks,
        seed=args.seed,
        profile=args.profile,
    )

    if use_wandb and "lighteval/math500" in metrics:
        import wandb
        wandb.log({"math500_pass@1": metrics["lighteval/math500"]})

    have = [metrics[k] for k in AVG5_KEYS if k in metrics and metrics[k] == metrics[k]]
    if len(have) == len(AVG5_KEYS):
        avg5 = sum(have) / len(have) * 100
        metrics["lighteval/avg5"] = avg5
        logger.info(f"[avg5] {avg5:.3f}")
        if use_wandb:
            import wandb
            wandb.log({"lighteval/avg5": avg5})
    else:
        logger.warning(f"avg5 not computed -- only {len(have)}/{len(AVG5_KEYS)} tasks produced a score")

    summary_path = os.path.join(args.model_dir, "eval_summary_resumed.json")
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"summary saved: {summary_path}")

    # ── HF upload (main.py's push_to_hub tail) ──────────────────────────────
    if not args.no_hub:
        repo = args.hub_repo
        if not repo:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            kd_tag = f"-kd{_fmt_float(args.kd_lambda)}" if args.kd_lambda > 0 else ""
            repo = (f"cosmos1030/gmp{kd_tag}-s{int(args.sparsity * 100)}pct-"
                    f"lr{_fmt_float(args.lr)}_{now}")
        logger.info(f"uploading to HuggingFace Hub: {repo}")
        try:
            for env in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                os.environ.pop(env, None)
            try:
                import huggingface_hub.constants as hf_const
                hf_const.HF_HUB_OFFLINE = False
            except Exception:
                pass
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=repo, exist_ok=True)
            api.upload_folder(
                folder_path=args.model_dir,
                repo_id=repo,
                # the lighteval output/cache dirs live inside the model dir; only ship weights
                ignore_patterns=["lighteval_bench/*", "*/lighteval|*", "eval_summary_resumed.json",
                                 ".zeroshot_done", ".eval_ctx.json"],
                commit_message=f"ELSA pruned (resumed eval): sparsity={args.sparsity}, lr={args.lr}",
            )
            url = f"https://huggingface.co/{repo}"
            logger.info(f"uploaded to {url}")
            if use_wandb:
                import wandb
                wandb.run.summary["hub_model_id"] = repo
                wandb.run.summary["hub_model_url"] = url
        except Exception as e:
            logger.warning(f"push_to_hub upload failed ({e}); model still on disk at {args.model_dir}")

    if use_wandb:
        import wandb
        wandb.finish()
    logger.info("=== resume_eval_lighteval.py done ===")
    for k in AVG5_KEYS + ["lighteval/avg5"]:
        if k in metrics:
            logger.info(f"  {k}: {metrics[k]}")


if __name__ == "__main__":
    main()
