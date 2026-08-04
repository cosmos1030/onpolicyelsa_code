"""
Build unified JSONL for GMP NTP+KD training:
  80% OpenThoughts3-1.2M  (chat CoT traces, prompt masked)
  20% FineWeb-Edu          (plain pretrain text, full gradient)

Output format per row:
  OpenThoughts3:
    {"text": "<chat-templated full conversation>"}
  FineWeb-Edu:
    {"text": "<raw plain text>", "pretrain": true}

Tokenization/chat-templating is parallelized across CPU cores via
datasets.Dataset.map(num_proc=...) — this is pure CPU work (no model forward
pass), so num_proc should track --cpus-per-task, not GPU count.

Usage:
    python scripts/build_ot3_fineweb_dataset.py \
        --nsamples 200000 \
        --out_path data/ot3_fineweb_200k.jsonl \
        --model_path <tokenizer path> \
        --num_proc 16
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

_ROLE_MAP = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}


def _render_ot(batch, tok, min_tokens):
    texts = []
    for convs in batch["conversations"]:
        if isinstance(convs, str):
            convs = json.loads(convs)
        msgs = [{"role": _ROLE_MAP.get(c["from"], c["from"]), "content": c["value"]} for c in convs]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            texts.append(None)
            continue
        texts.append(text)
    n_tokens = [
        len(tok(t, add_special_tokens=False).input_ids) if t is not None else 0
        for t in texts
    ]
    return {"text": texts, "n_tokens": n_tokens}


def _measure_fw(batch, tok):
    n_tokens = [len(tok(t, add_special_tokens=False).input_ids) for t in batch["text"]]
    return {"n_tokens": n_tokens}


def build(nsamples: int, out_path: str, model_path: str, seed: int = 42,
          min_tokens: int = 64, num_proc: int = 8):
    n_ot = int(nsamples * 0.8)
    n_fw = nsamples - n_ot

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    records = []

    # ── OpenThoughts3 ─────────────────────────────────────────────────────────
    print(f"Loading OpenThoughts3 (target {n_ot} samples)...", flush=True)
    ds_ot = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    ds_ot = ds_ot.shuffle(seed=seed).select(range(min(n_ot * 3, len(ds_ot))))
    ds_ot = ds_ot.map(
        lambda batch: _render_ot(batch, tok, min_tokens),
        batched=True, batch_size=256, num_proc=num_proc,
        remove_columns=[c for c in ds_ot.column_names if c != "conversations"],
        desc="Rendering OT3 chat template",
    )
    ds_ot = ds_ot.filter(lambda ex: ex["text"] is not None and ex["n_tokens"] >= min_tokens,
                         desc="Filtering OT3 by length")
    ot_records = [{"text": t} for t in ds_ot["text"][:n_ot]]
    print(f"  → {len(ot_records)} OT samples", flush=True)
    records.extend(ot_records)

    # ── FineWeb-Edu ───────────────────────────────────────────────────────────
    print(f"Loading FineWeb-Edu (target {n_fw} samples)...", flush=True)
    ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
    ds_fw = ds_fw.shuffle(seed=seed).select(range(min(n_fw * 3, len(ds_fw))))
    ds_fw = ds_fw.map(
        lambda batch: _measure_fw(batch, tok),
        batched=True, batch_size=256, num_proc=num_proc,
        desc="Measuring FineWeb-Edu length",
    )
    ds_fw = ds_fw.filter(lambda ex: ex["n_tokens"] >= min_tokens, desc="Filtering FineWeb by length")
    fw_records = [{"text": t, "pretrain": True} for t in ds_fw["text"][:n_fw]]
    print(f"  → {len(fw_records)} FineWeb samples", flush=True)
    records.extend(fw_records)

    import random
    random.Random(seed).shuffle(records)
    print(f"Total: {len(records)} samples. Writing to {out_path}...", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Done.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamples", type=int, default=20000)
    parser.add_argument("--out_path", default="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_20k.jsonl")
    parser.add_argument("--model_path",
                        default="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()
    build(args.nsamples, args.out_path, args.model_path, args.seed, args.min_tokens, args.num_proc)


if __name__ == "__main__":
    main()
