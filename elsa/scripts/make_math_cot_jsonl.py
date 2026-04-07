"""
Generate math_220k_cot.jsonl from open-r1/OpenR1-Math-220k.

Each row in the output has:
  {"text": "<problem>\n\n<generation>"}

where <generation> is the first entry in the 'generations' field
(a model-generated <think>...</think> reasoning trace).

Usage:
  python scripts/make_math_cot_jsonl.py \
      --output /home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl
"""

import ast
import json
import argparse
from datasets import load_dataset


def parse_generations(raw):
    """Parse the generations field, which is stored as a stringified Python list."""
    if isinstance(raw, list):
        return raw
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return [str(raw)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    print(f"Loading open-r1/OpenR1-Math-220k ({args.split})...")
    ds = load_dataset("open-r1/OpenR1-Math-220k", split=args.split)
    print(f"  {len(ds)} examples")

    skipped = 0
    with open(args.output, "w") as f:
        for row in ds:
            problem = (row.get("problem") or "").strip()
            gens = parse_generations(row.get("generations", []))

            # Take the first generation that is non-empty
            generation = ""
            for g in gens:
                g = g.strip()
                if g:
                    generation = g
                    break

            if not generation:
                skipped += 1
                continue

            text = f"{problem}\n\n{generation}" if problem else generation
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    total = len(ds) - skipped
    print(f"Written {total} rows to {args.output} ({skipped} skipped — empty generations)")


if __name__ == "__main__":
    main()
