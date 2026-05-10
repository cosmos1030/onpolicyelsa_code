"""Fast dataset stats: reads 5000 samples directly from JSONL, no full-dataset init."""
import sys, json, statistics
sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
from transformers import AutoTokenizer

MODEL = '/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
DATA  = '/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl'
N     = 5000
MAX_LEN = 2048
MAX_PROMPT_LEN = 512
THINK_TAG = "<think>"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
eos_id      = tokenizer.eos_token_id
think_close = tokenizer.encode('</think>', add_special_tokens=False)
im_end      = tokenizer.encode('<|im_end|>', add_special_tokens=False)

print(f"eos_token_id={eos_id}", flush=True)
print(f"</think> tokens={think_close}", flush=True)
print(f"<|im_end|> tokens={im_end}", flush=True)

# Read first N valid records
print(f"Reading {N} samples from JSONL...", flush=True)
records = []
with open(DATA) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        text = rec.get("text", "")
        if not text:
            continue
        records.append(text)
        if len(records) >= N:
            break

print(f"Got {len(records)} records, tokenizing...", flush=True)

stats = dict(total=0, truncated=0, has_eos=0, has_think_close=0, has_im_end=0, has_final_ans=0)
sup_lens = []

for i, text in enumerate(records):
    if i % 500 == 0:
        print(f"  {i}/{len(records)}...", flush=True)

    # Find prompt split
    idx = text.find(THINK_TAG)
    if idx == -1:
        idx = text.find("\n\n")
        if idx == -1:
            continue
        prompt_text = text[:idx]
    else:
        prompt_text = text[:idx + len(THINK_TAG)]

    # Tokenize full sequence
    full_ids = tokenizer(
        text, truncation=True, max_length=MAX_LEN,
        return_tensors="pt", padding=False,
    )["input_ids"].squeeze(0)

    # Tokenize prompt
    prompt_len = tokenizer(
        prompt_text, truncation=True, max_length=MAX_PROMPT_LEN,
        return_tensors="pt", padding=False,
    )["input_ids"].shape[1]

    # Supervised tokens = full_ids[prompt_len:]
    sup_list = full_ids[prompt_len:].tolist()

    stats['total'] += 1
    sup_lens.append(len(sup_list))

    if len(full_ids) >= MAX_LEN:
        stats['truncated'] += 1
    if eos_id in sup_list:
        stats['has_eos'] += 1
    for j in range(len(sup_list) - len(think_close) + 1):
        if sup_list[j:j+len(think_close)] == think_close:
            stats['has_think_close'] += 1
            break
    for j in range(len(sup_list) - len(im_end) + 1):
        if sup_list[j:j+len(im_end)] == im_end:
            stats['has_im_end'] += 1
            break
    decoded = tokenizer.decode(sup_list, skip_special_tokens=False)
    if r'\boxed' in decoded:
        stats['has_final_ans'] += 1

print(f"\n=== Results (N={stats['total']}) ===")
print(f"sup_len  median={statistics.median(sup_lens):.0f}  mean={sum(sup_lens)/len(sup_lens):.0f}  max={max(sup_lens)}", flush=True)
print()
for k, v in stats.items():
    pct = 100*v/stats['total'] if stats['total'] > 0 else 0
    print(f"  {k}: {v}/{stats['total']} ({pct:.1f}%)", flush=True)
print("Done.", flush=True)
