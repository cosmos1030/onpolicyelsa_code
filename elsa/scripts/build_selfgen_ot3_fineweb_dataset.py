"""
Build an ALPS calibration JSONL from self-gen CoT traces (dense Qwen3-1.7B's
OWN completions on the OT3 prompts extracted by extract_ot3_prompts.py, via
RAC's grpo.py --trace_only) mixed with FineWeb-Edu, replicating the same
80/20 OT/FW recipe and packing logic as build_ot3_fineweb_dataset.py -- the
only difference is the OT3 side is self-generated instead of using the
original OpenThoughts3 teacher's answer.

Usage:
    python scripts/build_selfgen_ot3_fineweb_dataset.py \
        --trace_path /home1/doyoonkim/projects/RAC/open-r1-main/math_trace/dataset_..._.jsonl \
        --out_path data/selfgen_ot3_fineweb_qwen3_8192.jsonl \
        --model_path <Qwen3-1.7B path> \
        --seqlen 8192
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from build_ot3_fineweb_dataset import _pack_fw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_path", required=True, help="Self-gen trace JSONL (prompt+completion columns) from RAC's grpo.py --trace_only.")
    ap.add_argument("--out_path", default="/home1/doyoonkim/projects/elsa/data/selfgen_ot3_fineweb_qwen3_8192.jsonl")
    ap.add_argument("--model_path",
                     default="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--ot_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_tokens", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=8192)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading self-gen traces from {args.trace_path}...", flush=True)
    records = []
    with open(args.trace_path) as f:
        for line in f:
            row = json.loads(line)
            prompt, completion = row.get("prompt"), row.get("completion")
            if not prompt or not completion:
                continue
            text = prompt + completion + "<|im_end|>\n"
            n_tok = len(tok(text, add_special_tokens=False).input_ids)
            if n_tok >= args.min_tokens:
                records.append({"text": text})
    n_ot = len(records)
    print(f"  → {n_ot} self-gen OT3 rows (>= {args.min_tokens} tok)", flush=True)

    n_fw = int(round(n_ot * (1 - args.ot_ratio) / args.ot_ratio))
    print(f"Loading FineWeb-Edu (target {n_fw} packed samples)...", flush=True)
    fw_pool_mult = max(3, (args.seqlen // 700) * 2)
    ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
    ds_fw = ds_fw.shuffle(seed=args.seed).select(range(min(n_fw * fw_pool_mult, len(ds_fw))))
    fw_records = _pack_fw(ds_fw["text"], tok, n_fw, args.seqlen)
    print(f"  → {len(fw_records)} FineWeb samples", flush=True)
    records.extend(fw_records)

    import random
    random.Random(args.seed).shuffle(records)
    print(f"Total: {len(records)} samples. Writing to {args.out_path}...", flush=True)

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
