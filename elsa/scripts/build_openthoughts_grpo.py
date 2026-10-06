"""
Build a GRPO-ready math dataset from OpenThoughts3-1.2M.

Unlike elsa's existing OT3 builds (ot3_fineweb_*.jsonl), which flatten each
row into a single pre-rendered chat "text" field for SFT/KD, GRPO needs the
prompt and a verifiable gold answer as SEPARATE columns so a reward function
can grade freshly-sampled rollouts against the gold answer.

OpenThoughts3-1.2M's raw schema is {difficulty, source, domain, conversations:
[{from: human, value: <problem>}, {from: gpt, value: <full CoT + final answer>}]}.
Shards are grouped by domain, not interleaved -- confirmed by sampling:
  shards 0-24   = code
  shards 25-109 = math   (85 shards x 10k rows = ~850k rows)
  shards 110-119 = science
This script only touches the math shard range.

Output JSONL columns:
  problem  -- conversations[0]['value'], the raw human turn
  solution -- just the final \\boxed{...} expression pulled out of the GPT
              turn's full CoT (not the whole CoT), so it matches the same
              "short gold string containing \\boxed{}" shape lighteval's
              MultilingualExtractiveMatchMetric expects when grading MATH-500
              elsewhere in this project. Rows where no \\boxed{} can be found
              in the teacher's answer are dropped (no verifiable gold to
              reward against).

Usage:
    python scripts/build_openthoughts_grpo.py \\
        --nsamples 20000 \\
        --out_path data/openthoughts3_math_grpo_20k.jsonl \\
        --seed 42
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

from datasets import load_dataset

MATH_SHARD_START = 25
MATH_SHARD_END = 109  # inclusive
TOTAL_SHARDS = 120

_BOXED_RE = re.compile(r"\\boxed\{")


def _extract_last_boxed(text):
    # Find the LAST \boxed{...}, handling nested braces (e.g. \boxed{\frac{1}{2}}).
    starts = [m.start() for m in _BOXED_RE.finditer(text)]
    if not starts:
        return None
    start = starts[-1]
    i = start + len("\\boxed{")
    depth = 1
    buf = []
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(c)
        i += 1
    if depth != 0:
        return None
    inner = "".join(buf)
    return f"\\boxed{{{inner}}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsamples", type=int, default=20000, help="target number of rows AFTER filtering for a valid \\boxed{} answer")
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_problem_chars", type=int, default=20)
    ap.add_argument("--max_problem_chars", type=int, default=4000)
    args = ap.parse_args()

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Each math shard has 10k raw rows; only download as many shards as needed
    # to comfortably cover --nsamples after boxed-answer filtering (rather
    # than all 85 math shards, ~20GB) -- 40% margin for dropped rows, min 3
    # shards for source diversity.
    rng = random.Random(args.seed)
    all_math_shards = list(range(MATH_SHARD_START, MATH_SHARD_END + 1))
    rng.shuffle(all_math_shards)
    # Empirically ~61-62% of rows get dropped for lacking a \boxed{} answer,
    # so a raw:kept ratio of ~2.8x (with margin) is needed, not 1.4x.
    n_shards_needed = max(3, math.ceil(args.nsamples / 10000 * 2.8))
    chosen_shards = sorted(all_math_shards[:n_shards_needed])
    data_files = [
        f"data/train-{i:05d}-of-{TOTAL_SHARDS:05d}.parquet"
        for i in chosen_shards
    ]
    print(f"Loading {len(data_files)} of {len(all_math_shards)} math shards (indices {chosen_shards}) from OpenThoughts3-1.2M...")
    ds = load_dataset(
        "open-thoughts/OpenThoughts3-1.2M",
        data_files={"train": data_files},
        split="train",
        verification_mode="no_checks",
    )
    print(f"Loaded {len(ds)} raw math rows, shuffling...")
    ds = ds.shuffle(seed=args.seed)

    n_written = 0
    n_no_boxed = 0
    n_bad_len = 0
    n_scanned = 0

    with open(out_path, "w") as f:
        for row in ds:
            n_scanned += 1
            if n_written >= args.nsamples:
                break
            convs = row["conversations"]
            if len(convs) < 2:
                continue
            human = next((c["value"] for c in convs if c["from"] == "human"), None)
            gpt = next((c["value"] for c in convs if c["from"] == "gpt"), None)
            if not human or not gpt:
                continue
            if not (args.min_problem_chars <= len(human) <= args.max_problem_chars):
                n_bad_len += 1
                continue
            boxed = _extract_last_boxed(gpt)
            if boxed is None:
                n_no_boxed += 1
                continue
            f.write(json.dumps({
                "problem": human,
                "solution": boxed,
                "difficulty": row.get("difficulty"),
                "source": row.get("source"),
            }) + "\n")
            n_written += 1
            if n_written % 2000 == 0:
                print(f"  written {n_written}/{args.nsamples} (scanned {n_scanned})")

    print(f"Done. Wrote {n_written} rows to {out_path} (scanned {n_scanned}, "
          f"dropped {n_no_boxed} no-boxed, {n_bad_len} bad-length).")


if __name__ == "__main__":
    main()
