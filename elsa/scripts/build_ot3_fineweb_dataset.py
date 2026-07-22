"""
Build unified JSONL for GMP NTP+KD training:
  80% OpenThoughts3-1.2M  (chat CoT traces, prompt masked)
  20% FineWeb-Edu          (plain pretrain text, full gradient)

Output format per row:
  OpenThoughts3:
    {"text": "<chat-templated full conversation>"}
  FineWeb-Edu:
    {"text": "<raw plain text>", "pretrain": true}

Usage:
    python scripts/build_ot3_fineweb_dataset.py \
        --nsamples 200000 \
        --out_path data/ot3_fineweb_200k.jsonl \
        --model_path <tokenizer path>
"""

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def build(nsamples: int, out_path: str, model_path: str, seed: int = 42, min_tokens: int = 64):
    rng = random.Random(seed)
    n_ot = int(nsamples * 0.8)
    n_fw = nsamples - n_ot

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

    records = []

    # ── OpenThoughts3 ─────────────────────────────────────────────────────────
    print(f"Loading OpenThoughts3 (target {n_ot} samples)...", flush=True)
    ds_ot = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    indices = rng.sample(range(len(ds_ot)), min(n_ot * 3, len(ds_ot)))

    ot_records = []
    for i in indices:
        if len(ot_records) >= n_ot:
            break
        ex = ds_ot[i]
        convs = ex["conversations"]
        if isinstance(convs, str):
            import json as _json
            convs = _json.loads(convs)
        msgs = [{"role": _role_map.get(c["from"], c["from"]), "content": c["value"]} for c in convs]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            continue
        if len(tok(text, add_special_tokens=False).input_ids) < min_tokens:
            continue
        ot_records.append({"text": text})

    print(f"  → {len(ot_records)} OT samples", flush=True)
    records.extend(ot_records)

    # ── FineWeb-Edu ───────────────────────────────────────────────────────────
    print(f"Loading FineWeb-Edu (target {n_fw} samples)...", flush=True)
    ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
    fw_indices = rng.sample(range(len(ds_fw)), min(n_fw * 3, len(ds_fw)))

    fw_records = []
    for i in fw_indices:
        if len(fw_records) >= n_fw:
            break
        text = ds_fw[i]["text"]
        if len(tok(text, add_special_tokens=False).input_ids) < min_tokens:
            continue
        fw_records.append({"text": text, "pretrain": True})

    print(f"  → {len(fw_records)} FineWeb samples", flush=True)
    records.extend(fw_records)

    rng.shuffle(records)
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
    args = parser.parse_args()
    build(args.nsamples, args.out_path, args.model_path, args.seed, args.min_tokens)


if __name__ == "__main__":
    main()
