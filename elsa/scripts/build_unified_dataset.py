"""
Build math_220k_unified.jsonl from OpenR1-Math-220k HF cache.

Output format per row:
  {
    "problem": "<raw question text>",
    "prompt":  "<BOS><User>problem<Assistant><think>\n",   # for IPO/OPKD generation
    "text":    "<BOS><User>problem<Assistant><think>\nCoT</think>\n\nAnswer",  # for NTP training
    "answer":  "<answer string>",
  }

MixedTextDataset splits 'text' at '<think>' → prompt_ids / cot_ids (unchanged logic).
MixedPromptDataset reads 'prompt' field directly (unchanged logic).
"""
import json
import os
import sys
from pathlib import Path

import datasets
from transformers import AutoTokenizer

MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
HF_CACHE   = "/home1/doyoonkim/.cache/huggingface/hub"
OUT_PATH   = "/home1/doyoonkim/projects/elsa/data/math_220k_unified.jsonl"
THINK_PREFIX = "<think>\n"

def main():
    print("Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading OpenR1-Math-220k...", flush=True)
    ds = datasets.load_dataset(
        "open-r1/OpenR1-Math-220k",
        split="train",
        cache_dir=HF_CACHE,
    )
    print(f"Dataset rows: {len(ds)}", flush=True)

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skip    = 0

    with open(out_path, "w") as fout:
        for i, row in enumerate(ds):
            problem    = row["problem"]
            generations = row["generations"]   # list of strings
            answer     = row.get("answer", "")

            if not generations:
                n_skip += 1
                continue

            gen = generations[0]  # first (best) generation
            if not gen.startswith(THINK_PREFIX):
                # unexpected format — skip
                n_skip += 1
                continue

            # prompt: BOS + User + problem + Assistant + <think>\n
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": problem}],
                tokenize=False,
                add_generation_prompt=True,
            )
            # prompt ends with "<think>\n"; verify
            assert prompt.endswith(THINK_PREFIX), f"unexpected prompt tail: {repr(prompt[-30:])}"

            # text: prompt + rest of generation (skip duplicate "<think>\n" prefix)
            text = prompt + gen[len(THINK_PREFIX):]

            record = {
                "problem": problem,
                "prompt":  prompt,
                "text":    text,
                "answer":  answer,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

            if (i + 1) % 10000 == 0:
                print(f"  {i+1}/{len(ds)} rows processed, written={n_written}, skipped={n_skip}", flush=True)

    print(f"Done. Written={n_written}, Skipped={n_skip} → {OUT_PATH}", flush=True)

if __name__ == "__main__":
    main()
