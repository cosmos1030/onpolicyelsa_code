import json
import random
import re
import sys

import pyarrow.parquet as pq
from vllm import LLM, SamplingParams

from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.utils.language import Language
import types

MODEL = sys.argv[1] if len(sys.argv) > 1 else "cosmos1030/gmp-kd3e-1-s70pct-lr5e-5_20260811_115604"
N_QUESTIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
N_SAMPLES = int(sys.argv[3]) if len(sys.argv) > 3 else 8
OUT_JSON = sys.argv[4] if len(sys.argv) > 4 else "/home1/doyoonkim/projects/elsa/models/multisample_s70.json"

SRC_PARQUET = "/home1/doyoonkim/projects/elsa/models/qwen3_4b_trgmp_s70pct_lr5e-05_mi16_job41346_savedetails/details/cosmos1030/gmp-kd3e-1-s70pct-lr5e-5_20260811_115604/2026-08-21T18-47-47.291504/details_lighteval|math_500|0_2026-08-21T18-47-47.291504.parquet"

t = pq.read_table(SRC_PARQUET)
rows = t.to_pylist()
random.seed(0)
sample_rows = random.sample(rows, min(N_QUESTIONS, len(rows)))

prompts = []
qids = []
golds = []
for row in sample_rows:
    qids.append(row["doc"]["id"])
    prompts.append(row["model_response"]["input"])
    golds.append(row["doc"]["choices"][0])

print(f"Loaded {len(prompts)} prompts. Loading model {MODEL}...", flush=True)

llm = LLM(model=MODEL, dtype="bfloat16", trust_remote_code=True, tensor_parallel_size=1,
          gpu_memory_utilization=0.95, max_model_len=8192, swap_space=8)

sp = SamplingParams(n=N_SAMPLES, temperature=0.6, top_p=0.95, max_tokens=8192)

print("Generating...", flush=True)
outputs = llm.generate(prompts, sp)
print("Generation done, scoring...", flush=True)

GOLD_TARGET = (ExprExtractionConfig(),)
PRED_TARGET = (ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0))
FakeDoc = types.SimpleNamespace(choices=[])
GOLD_REGEXES = get_extraction_regexes(FakeDoc, GOLD_TARGET, Language.ENGLISH)
PRED_REGEXES = get_extraction_regexes(FakeDoc, PRED_TARGET, Language.ENGLISH)


def norm_bare(s):
    s = s.strip().rstrip(".").strip()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\left", "").replace("\\right", "").replace(" ", "").replace("$", "")
    return s.lower()


def fallback_bare_answer_match(pred_text, gold_text):
    # No \boxed{} in the completion (common when a sampled generation drifts off the
    # required output format). Look for "answer is <value>" near the end of the text
    # and compare loosely against the gold's own boxed value.
    gold_boxed = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", gold_text)
    if not gold_boxed:
        return False
    gold_val = norm_bare(gold_boxed[-1])
    tail = pred_text[-400:]
    candidates = re.findall(r"(?:answer is|answer:|=)\s*\*{0,2}([^\n.]{1,80})", tail, flags=re.IGNORECASE)
    for c in reversed(candidates):
        if norm_bare(c).startswith(gold_val) or gold_val == norm_bare(c):
            return True
    return False


def is_correct(pred_text, gold_text, debug=False):
    extracted_pred = extract_target_from_pred(pred_text, PRED_REGEXES, "first_match", "any_match", 5)
    extracted_gold = extract_target_from_pred(gold_text, GOLD_REGEXES, "first_match", "any_match", 5)
    if not extracted_gold:
        extracted_gold = [gold_text]
    if not extracted_pred:
        result = fallback_bare_answer_match(pred_text, gold_text)
        if debug:
            print(f"  [DEBUG] no \\boxed pred; fallback bare-match -> {result}. pred_text tail: {pred_text[-200:]!r}", flush=True)
        return result
    result = compare_gold_target(extracted_gold, extracted_pred, 5, timeout_seconds=5)
    if debug:
        print(f"  [DEBUG] gold={extracted_gold} pred={extracted_pred} -> {result}", flush=True)
    return result


results = []
for qi, (qid, gold, out) in enumerate(zip(qids, golds, outputs)):
    entry = {"qid": qid, "samples": []}
    for ci, comp in enumerate(out.outputs):
        text = comp.text
        length = len(comp.token_ids)
        dbg = True  # small debug run: print everything
        correct = is_correct(text, gold, debug=dbg)
        entry["samples"].append({"len": length, "correct": correct, "text": text if dbg else None})
    results.append(entry)
    if len(results) % 20 == 0:
        print(f"scored {len(results)}/{len(qids)}", flush=True)

with open(OUT_JSON, "w") as f:
    json.dump(results, f)

print(f"Saved to {OUT_JSON}", flush=True)

# quick summary
all_correct_lens = []
all_wrong_lens = []
for entry in results:
    for s in entry["samples"]:
        (all_correct_lens if s["correct"] else all_wrong_lens).append(s["len"])
n_c, n_w = len(all_correct_lens), len(all_wrong_lens)
avg_c = sum(all_correct_lens) / max(1, n_c)
avg_w = sum(all_wrong_lens) / max(1, n_w)
print(f"n_correct={n_c} avg_len={avg_c:.1f}  n_wrong={n_w} avg_len={avg_w:.1f}")
