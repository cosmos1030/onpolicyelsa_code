"""
SparseGPT pruning + full evaluation pipeline.

Runs in order:
  1. SparseGPT prune (OpenThoughts3 80% + FineWeb-Edu 20%)
  2. WikiText2 / C4 PPL
  3. Zero-shot tasks (boolq, rte, hellaswag, winogrande, arc_easy,
                      arc_challenge, openbookqa, piqa, race)
  4. lighteval benchmarks via elsa/lib/lighteval_bench.py (shared with the
     ELSA/GMP/ALPS/SparseLLM eval pipelines): MATH-500, GPQA-Diamond, IFEval,
     LiveCodeBench, GSM8K

Usage:
    python prune_and_eval.py \
        --model_path <path> \
        --sparsity 0.5 \
        --nsamples 128 \
        --seqlen 2048 \
        --save_path <out> \
        --wandb_project reasoning_pruning_v1 \
        --wandb_name sparsegpt_s50_n128

    # Smoke test (very fast, just checks pipeline):
    python prune_and_eval.py ... --smoketest
"""

import argparse
import json
import logging
import os
import random
import sys

import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ELSA_PATH = "/home1/doyoonkim/projects/elsa"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── calibration data ────────────────────────────────────────────────────────

def load_calib_jsonl(path: str, nsamples: int, seqlen: int, seed: int = 42) -> Dataset:
    """Calibration set from a prebuilt JSONL with a "text" column.

    Used for the self-gen recipe: the OT3 side is the model's OWN completions
    on the OT3 prompts rather than the original teacher's answers, already
    mixed 80/20 with FineWeb-Edu and packed by
    elsa/scripts/build_selfgen_ot3_fineweb_dataset.py -- the same files ALPS
    calibrates on, so ALPS and SparseGPT numbers stay directly comparable.

    No length filtering here (unlike build_calib_dataset): these files are
    pre-packed to a fixed length, and dropping rows shorter than `seqlen`
    would silently shrink an already-small (~127-row) calibration set. Rows
    are shuffled and truncated to nsamples; if the file has fewer rows than
    nsamples, every row is used and a warning is logged, since the effective
    calibration budget is then smaller than the caller asked for.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append({"text": json.loads(line)["text"]})
    if not rows:
        raise ValueError(f"[calib] no rows read from {path}")
    random.Random(seed).shuffle(rows)
    if len(rows) < nsamples:
        logger.warning(
            f"[calib] {path} has only {len(rows)} rows < nsamples={nsamples}; "
            f"using all {len(rows)} (effective calibration budget is smaller than requested)")
    rows = rows[:nsamples]
    logger.info(f"[calib] self-gen JSONL {path}: {len(rows)} samples")
    return Dataset.from_list(rows)


def build_calib_dataset(tokenizer, nsamples: int, seqlen: int, seed: int = 42) -> Dataset:
    """OpenThoughts3 80% + FineWeb-Edu 20%."""
    rng = random.Random(seed)
    n_ot = max(1, int(nsamples * 0.8))
    n_fw = nsamples - n_ot

    logger.info(f"[calib] Loading OpenThoughts3 ({n_ot} samples)...")
    raw_ot = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    _role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
    ot_samples = []
    indices = rng.sample(range(len(raw_ot)), min(n_ot * 10, len(raw_ot)))
    for i in indices:
        if len(ot_samples) >= n_ot:
            break
        ex = raw_ot[i]
        convs = ex["conversations"]
        if isinstance(convs, str):
            convs = json.loads(convs)
        msgs = [{"role": _role_map.get(c["from"], c["from"]), "content": c["value"]} for c in convs]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            continue
        if len(tokenizer(text, add_special_tokens=False).input_ids) >= seqlen:
            ot_samples.append({"text": text})
    logger.info(f"[calib]   → {len(ot_samples)} OT samples")

    logger.info(f"[calib] Loading FineWeb-Edu ({n_fw} samples)...")
    raw_fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
    fw_samples = []
    fw_idx = rng.sample(range(len(raw_fw)), min(n_fw * 10, len(raw_fw)))
    for i in fw_idx:
        if len(fw_samples) >= n_fw:
            break
        text = raw_fw[i]["text"]
        if len(tokenizer(text, add_special_tokens=False).input_ids) >= seqlen:
            fw_samples.append({"text": text})
    logger.info(f"[calib]   → {len(fw_samples)} FW samples")

    all_samples = ot_samples + fw_samples
    rng.shuffle(all_samples)
    all_samples = all_samples[:nsamples]
    logger.info(f"[calib] Total: {len(all_samples)} samples")
    return Dataset.from_list(all_samples)


# ─── PPL eval ────────────────────────────────────────────────────────────────

def eval_ppl(model, tokenizer, seqlen: int) -> dict:
    sys.path.insert(0, ELSA_PATH)
    from lib.eval import calculate_ppl
    from lib.data import get_loaders

    model.seqlen = seqlen
    model.eval()
    results = {}
    for ds_name in ["wikitext2", "c4"]:
        logger.info(f"[ppl] Evaluating {ds_name}...")
        _, testloader = get_loaders(ds_name, seed=42, seqlen=seqlen, tokenizer=tokenizer)
        with torch.no_grad():
            ppl = calculate_ppl(model, testloader, tokenizer, bs=1)
        results[f"ppl/{ds_name}"] = ppl
        logger.info(f"[ppl] {ds_name} = {ppl:.4f}")
    return results


# ─── zero-shot eval ──────────────────────────────────────────────────────────

ZERO_SHOT_TASKS = [
    "boolq", "rte", "hellaswag", "winogrande",
    "arc_easy", "arc_challenge", "openbookqa", "piqa", "race",
]

def eval_zero_shot(model, tokenizer, limit=None) -> dict:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    logger.info(f"[zero_shot] Evaluating {ZERO_SHOT_TASKS}, limit={limit}")
    lm = HFLM(model)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=ZERO_SHOT_TASKS,
        num_fewshot=0,
        batch_size="auto",
        device=model.device,
        limit=limit,
        cache_requests=False,
        random_seed=42,
    )["results"]

    out = {}
    for task, vals in results.items():
        acc = vals.get("acc_norm,none", vals.get("acc,none"))
        if acc is not None:
            out[f"zero_shot/{task}"] = acc
            logger.info(f"[zero_shot] {task}: {acc:.4f}")
    if out:
        out["zero_shot/avg"] = sum(out.values()) / len(out)
    return out


# ─── lighteval benchmarks ─────────────────────────────────────────────────────
# Reuses the shared elsa/lib/lighteval_bench.py implementation (math500, gpqa,
# ifeval, lcb, gsm8k) instead of a separately-maintained 4-bench copy — was
# previously the only baseline missing gsm8k that the other baselines
# (ELSA/GMP via main.py, ALPS/SparseLLM via eval_full.py) get.

def eval_bench(model_path: str, out_base: str, gpu_util: float,
              max_samples: int | None = None) -> dict:
    if ELSA_PATH not in sys.path:
        sys.path.insert(0, ELSA_PATH)
    from lib.lighteval_bench import run_lighteval_bench
    return run_lighteval_bench(model_path, out_base, gpu_util=gpu_util,
                               max_samples=max_samples, log_to_wandb=False)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--prune_n", type=int, default=0)
    parser.add_argument("--prune_m", type=int, default=0)
    parser.add_argument("--calib_data_path", type=str, default=None,
                        help="Prebuilt calibration JSONL with a \"text\" column (self-gen recipe). "
                             "When omitted, calibration is built from HuggingFace as "
                             "OpenThoughts3 80%% + FineWeb-Edu 20%% (the original behavior).")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--scope", default="all", choices=["all", "mlp"])
    parser.add_argument("--wandb_project", default="reasoning_pruning_v1")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--method", default="sparsegpt", choices=["sparsegpt", "wanda"],
                        help="Pruning method: sparsegpt (default) or wanda")
    parser.add_argument("--skip_prune", action="store_true",
                        help="Skip pruning (eval-only mode for dense baseline)")
    parser.add_argument("--smoketest", action="store_true",
                        help="Fast run: nsamples=4, lm-eval limit=4, lighteval max-samples=2")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload pruned model to HuggingFace Hub after saving")
    parser.add_argument("--hub_model_id", type=str, default=None, help="HF Hub repo id (e.g. username/model-name); auto-generated if not given")
    args = parser.parse_args()

    if args.smoketest:
        args.nsamples = 4
        zs_limit = 4
        le_samples = 2
        logger.info("=== SMOKE TEST MODE ===")
    else:
        zs_limit = None
        le_samples = None

    # wandb init
    run_name = args.wandb_name or (
        f"{args.method}_s{int(args.sparsity*100)}_n{args.nsamples}"
        + ("_smoketest" if args.smoketest else "")
    )
    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "model_path": args.model_path,
            "method": args.method,
            "sparsity": args.sparsity,
            "nsamples": args.nsamples,
            "seqlen": args.seqlen,
            "prune_n": args.prune_n,
            "prune_m": args.prune_m,
            "calib_data": (args.calib_data_path if args.calib_data_path
                           else "openthoughts3_80pct_fineweb_20pct"),
            "smoketest": args.smoketest,
        },
    )

    # ── 1. Load model ─────────────────────────────────────────────────────────
    logger.info(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()

    # ── 2. SparseGPT pruning ──────────────────────────────────────────────────
    model = model.to("cuda")

    if args.skip_prune:
        logger.info("Skipping SparseGPT pruning (dense baseline eval mode)")
        # keep args.save_path as provided (used for lighteval output dir)
        eval_model_path = args.model_path
    else:
        # One-shot calibration is forward-only: ~2*N*tokens (no backward pass),
        # unlike gradient fine-tuning's ~6*N*tokens.
        n_params = sum(p.numel() for p in model.parameters())
        n_tokens = args.nsamples * args.seqlen
        flops = 2 * n_params * n_tokens
        logger.info(f"[calib] FLOPs: {flops:.3e} ({n_params} params x {n_tokens} tokens, forward-only)")
        wandb.log({"flops": flops})
        if args.calib_data_path:
            logger.info(f"Loading calibration dataset from {args.calib_data_path} (nsamples={args.nsamples})...")
            calib_dataset = load_calib_jsonl(args.calib_data_path, args.nsamples, args.seqlen, args.seed)
        else:
            logger.info(f"Building calibration dataset (nsamples={args.nsamples})...")
            calib_dataset = build_calib_dataset(tokenizer, args.nsamples, args.seqlen, args.seed)

        _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/open_r1 → src/
        if _src_dir not in sys.path:
            sys.path.insert(0, _src_dir)
        from open_r1.open_r1_trl.trl.pruner.pruning import make_calib_loader, sparsegpt_prune, prune_wanda

        calib_loader = make_calib_loader(
            calib_dataset, tokenizer,
            tokens=args.nsamples * args.seqlen,
            batch_size=1, prompt_column="text",
        )

        if args.method == "wanda":
            logger.info(f"Wanda: sparsity={args.sparsity}")
            prune_wanda(
                model, calib_loader,
                sparsity=args.sparsity,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            prunen = args.prune_n if args.prune_n > 0 else None
            prunem = args.prune_m if args.prune_m > 0 else None
            logger.info(f"SparseGPT: sparsity={args.sparsity}, N:M={prunen}:{prunem}")
            sparsegpt_prune(
                model, calib_loader,
                sparsity=args.sparsity,
                prunen=prunen, prunem=prunem,
                device="cuda" if torch.cuda.is_available() else "cpu",
                scope=args.scope,
            )

        logger.info(f"Saving pruned model to {args.save_path}")
        os.makedirs(args.save_path, exist_ok=True)
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        eval_model_path = args.save_path

        if args.push_to_hub:
            try:
                from huggingface_hub import HfApi
                for _env in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                    os.environ.pop(_env, None)
                try:
                    import huggingface_hub.constants as _hf_const
                    _hf_const.HF_HUB_OFFLINE = False
                except Exception:
                    pass
                hub_model_id = args.hub_model_id
                if not hub_model_id:
                    from datetime import datetime as _dt
                    _now = _dt.now().strftime("%Y%m%d_%H%M%S")
                    hub_model_id = f"cosmos1030/{args.method}-s{int(args.sparsity * 100)}pct_{_now}"
                logger.info(f"Uploading model to HuggingFace Hub: {hub_model_id}")
                api = HfApi()
                api.create_repo(repo_id=hub_model_id, exist_ok=True)
                # save_path doubles as the lighteval eval-cache dir later in this
                # script, so on a rerun against an already-evaluated save_path it
                # can contain leftover cache junk alongside the real model files —
                # only upload the actual HF model/tokenizer files.
                api.upload_folder(
                    folder_path=args.save_path,
                    repo_id=hub_model_id,
                    allow_patterns=[
                        "config.json", "generation_config.json", "chat_template.jinja",
                        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                        "vocab.json", "merges.txt", "tokenizer.json",
                        "model.safetensors", "model.safetensors.index.json", "model-*.safetensors",
                    ],
                    commit_message=f"{args.method} pruned: sparsity={args.sparsity}",
                )
                hub_url = f"https://huggingface.co/{hub_model_id}"
                logger.info(f"Uploaded to {hub_url}")
                wandb.run.summary["hub_model_id"] = hub_model_id
                wandb.run.summary["hub_model_url"] = hub_url
            except Exception as e:
                logger.warning(f"push_to_hub failed ({e}); continuing without upload.")

    # ── 3. PPL eval ───────────────────────────────────────────────────────────
    ppl_metrics = eval_ppl(model, tokenizer, args.seqlen)
    wandb.log(ppl_metrics)

    # ── 4. Zero-shot eval ────────────────────────────────────────────────────
    zs_metrics = eval_zero_shot(model, tokenizer, limit=zs_limit)
    wandb.log(zs_metrics)

    # Free GPU before lighteval+vLLM
    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # ── 5. lighteval benchmarks (math500/gpqa/ifeval/lcb/gsm8k) ──
    free, total = torch.cuda.mem_get_info(0)
    # Cap at 0.85 to leave headroom for MMLU LOGPROBS after prior GPU usage
    gpu_util = min(free / total * 0.92, 0.85)

    os.makedirs(args.save_path, exist_ok=True)
    out_base = os.path.join(args.save_path, "lighteval")
    le_metrics = eval_bench(eval_model_path, out_base, gpu_util, le_samples)
    wandb.log(le_metrics)

    all_metrics = {**ppl_metrics, **zs_metrics, **le_metrics}
    logger.info("=== Final metrics ===")
    for k, v in sorted(all_metrics.items()):
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    run.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()
