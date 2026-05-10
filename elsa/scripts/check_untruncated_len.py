"""Check actual untruncated CoT length distribution (no max_len cap)."""
import sys, json, statistics
sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
from transformers import AutoTokenizer

MODEL = '/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
DATA  = '/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl'
N     = 5000
MAX_PROMPT_LEN = 512
THINK_TAG = "<think>"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

records = []
with open(DATA) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        text = rec.get("text", "")
        if not text: continue
        records.append(text)
        if len(records) >= N: break

print(f"Got {len(records)} records, tokenizing (no truncation)...", flush=True)

full_lens = []
for i, text in enumerate(records):
    if i % 500 == 0:
        print(f"  {i}/{len(records)}...", flush=True)
    ids = tokenizer(text, truncation=False, return_tensors="pt", padding=False)["input_ids"].squeeze(0)
    full_lens.append(len(ids))

full_lens.sort()
n = len(full_lens)
thresholds = [2048, 4096, 8192, 16384]

print(f"\n=== Untruncated full sequence length (N={n}) ===")
print(f"  median={statistics.median(full_lens):.0f}  mean={sum(full_lens)/n:.0f}  max={max(full_lens)}  min={min(full_lens)}")
print()
for t in thresholds:
    fits = sum(1 for l in full_lens if l <= t)
    print(f"  <= {t:6d} tokens: {fits}/{n} ({100*fits/n:.1f}%)")
print("Done.", flush=True)
