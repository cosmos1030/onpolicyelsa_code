"""
Extract raw user-turn prompts from the exact same OpenThoughts3-1.2M
selection used by build_ot3_fineweb_dataset.py (same seed/shuffle/filter),
so a self-gen trace-generation pass (RAC's grpo.py --trace_only) can be run
on prompts as close as possible to the ones ALPS's OT80/FW20 calibration
data was actually built from -- just the question, not the original
teacher's answer.

Only pulls the first N_OT survivors of the same filter pipeline (not the
full 32000-example pool used by the production OT80/FW20 build), since
self-gen tracing is far more expensive than reusing a pre-existing dataset
and ALPS calibration only ever draws --nsamples=128 random windows anyway.

Usage:
    python scripts/extract_ot3_prompts.py --n_ot 200 --out_path data/ot3_prompts_200_qwen3.jsonl
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from build_ot3_fineweb_dataset import _render_ot, _ROLE_MAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ot", type=int, default=200, help="Number of OT3 prompts to extract.")
    ap.add_argument("--out_path", default="/home1/doyoonkim/projects/elsa/data/ot3_prompts_200_qwen3.jsonl")
    ap.add_argument("--model_path",
                     default="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_tokens", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--num_proc", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading OpenThoughts3 (target {args.n_ot} prompts)...", flush=True)
    ds_ot = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    ds_ot = ds_ot.shuffle(seed=args.seed).select(range(min(args.n_ot * 3, len(ds_ot))))
    ds_ot = ds_ot.map(
        lambda batch: _render_ot(batch, tok, args.min_tokens, seqlen=args.seqlen, strip_think=False),
        batched=True, batch_size=256, num_proc=args.num_proc,
        remove_columns=[c for c in ds_ot.column_names if c != "conversations"],
        desc="Rendering OT3 chat template (same pipeline as build_ot3_fineweb_dataset.py)",
    )
    ds_ot = ds_ot.filter(lambda ex: ex["text"] is not None and ex["n_tokens"] >= args.min_tokens,
                         desc="Filtering OT3 by length")

    n_avail = min(args.n_ot, len(ds_ot))
    print(f"  → {n_avail} surviving rows (requested {args.n_ot})", flush=True)

    records = []
    for convs in ds_ot["conversations"][:n_avail]:
        if isinstance(convs, str):
            convs = json.loads(convs)
        user_msg = next((c["value"] for c in convs if _ROLE_MAP.get(c["from"], c["from"]) == "user"), None)
        if user_msg:
            records.append({"prompt": user_msg})

    print(f"  → {len(records)} prompts extracted. Writing to {args.out_path}...", flush=True)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
