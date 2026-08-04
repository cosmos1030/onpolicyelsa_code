"""
Standalone SparseGPT pruning script with OpenThoughts3+FineWeb-Edu calibration.

Usage:
    python src/open_r1/prune_sparsegpt.py \
        --model_path <path> \
        --sparsity 0.5 \
        --prune_n 0 --prune_m 0 \
        --nsamples 128 \
        --seqlen 2048 \
        --save_path <output_path>
"""

import argparse
import json
import logging
import os
import random
import sys

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_openthoughts_fineweb_dataset(tokenizer, nsamples: int, seqlen: int, seed: int = 42) -> Dataset:
    """OpenThoughts3 80% + FineWeb-Edu 20% calibration dataset.

    Matches the ReasoningQAT paper calibration setup.
    Returns a Dataset with 'text' column of chat-templated strings.
    """
    rng = random.Random(seed)
    n_ot = max(1, int(nsamples * 0.8))
    n_fw = nsamples - n_ot

    # --- OpenThoughts3 (chat format → apply template) ---
    logger.info(f"Loading OpenThoughts3 ({n_ot} samples)...")
    raw_ot = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    _role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

    samples = []
    indices = rng.sample(range(len(raw_ot)), min(n_ot * 5, len(raw_ot)))
    for i in indices:
        if len(samples) >= n_ot:
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
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) >= seqlen:
            samples.append({"text": text})

    logger.info(f"  → got {len(samples)} OT samples")

    # --- FineWeb-Edu (plain text) ---
    logger.info(f"Loading FineWeb-Edu ({n_fw} samples)...")
    raw_fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
    fw_samples = []
    fw_indices = rng.sample(range(len(raw_fw)), min(n_fw * 5, len(raw_fw)))
    for i in fw_indices:
        if len(fw_samples) >= n_fw:
            break
        text = raw_fw[i]["text"]
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) >= seqlen:
            fw_samples.append({"text": text})

    logger.info(f"  → got {len(fw_samples)} FineWeb-Edu samples")

    all_samples = samples + fw_samples
    rng.shuffle(all_samples)
    all_samples = all_samples[:nsamples]
    logger.info(f"Total calibration samples: {len(all_samples)}")
    return Dataset.from_list(all_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--prune_n", type=int, default=0, help="N in N:M (0 = unstructured)")
    parser.add_argument("--prune_m", type=int, default=0, help="M in N:M (0 = unstructured)")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--scope", default="all", choices=["all", "mlp"])
    args = parser.parse_args()

    logger.info(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    )
    model.eval()

    calib_dataset = build_openthoughts_fineweb_dataset(tokenizer, args.nsamples, args.seqlen, args.seed)

    from open_r1_trl.trl.pruner.pruning import make_calib_loader, sparsegpt_prune

    calib_loader = make_calib_loader(
        calib_dataset,
        tokenizer,
        tokens=args.nsamples * args.seqlen,
        batch_size=1,
        prompt_column="text",
    )

    prunen = args.prune_n if args.prune_n > 0 else None
    prunem = args.prune_m if args.prune_m > 0 else None

    logger.info(
        f"SparseGPT pruning: sparsity={args.sparsity}, N:M={prunen}:{prunem}, scope={args.scope}"
    )
    sparsegpt_prune(
        model,
        calib_loader,
        sparsity=args.sparsity,
        prunen=prunen,
        prunem=prunem,
        device="cuda" if torch.cuda.is_available() else "cpu",
        scope=args.scope,
    )

    logger.info(f"Saving pruned model to {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
