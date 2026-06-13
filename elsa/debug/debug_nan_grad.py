"""
Diagnose NaN gradient location with flash_attention_2 + batch_size=2.
Runs 1 forward+backward and prints the first parameter with NaN grad.
"""
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DATA_PATH  = "/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

print("Loading model (flash_attention_2)...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
print(f"pad_token_id={tok.pad_token_id}, eos_token_id={tok.eos_token_id}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).cuda()
model.train()
model.config.use_cache = False

class TinyDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, n=200):
        self.samples = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= n: break
                text = json.loads(line).get("text", "")
                enc = tokenizer(text, max_length=max_len, truncation=True, return_tensors="pt")
                ids = enc["input_ids"][0]
                if len(ids) < 4: continue
                self.samples.append({
                    "input_ids": ids,
                    "attention_mask": enc["attention_mask"][0],
                    "labels": ids.clone(),
                })
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id  = tok.pad_token_id
    result  = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        pad = max_len - b["input_ids"].shape[0]
        result["input_ids"].append(
            torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
        result["attention_mask"].append(
            torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        result["labels"].append(
            torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
    return {k: torch.stack(v) for k, v in result.items()}

ds = TinyDataset(DATA_PATH, tok, max_len=512)
loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate)

batch = next(iter(loader))
print(f"input_ids shape: {batch['input_ids'].shape}")
print(f"attention_mask sum per row: {batch['attention_mask'].sum(dim=1).tolist()}")
print(f"input_ids[0,:5]={batch['input_ids'][0,:5].tolist()}, input_ids[1,:5]={batch['input_ids'][1,:5].tolist()}")
batch = {k: v.cuda() for k, v in batch.items()}

print("\n--- Forward pass ---")
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out = model(**batch)
    loss = out.loss
print(f"loss={loss.item():.4f}, is_finite={torch.isfinite(loss).item()}")

print("\n--- Backward pass ---")
loss.backward()

print("\n--- Scanning for NaN grads ---")
nan_found = False
for name, p in model.named_parameters():
    if p.grad is None:
        continue
    if not torch.isfinite(p.grad).all():
        bad_count = (~torch.isfinite(p.grad)).sum().item()
        print(f"  NaN/Inf grad: {name}  dtype={p.grad.dtype}  shape={tuple(p.grad.shape)}  "
              f"bad_elements={bad_count}  max_abs={p.grad.abs().max().item():.4e}")
        nan_found = True
        if bad_count > 0:
            break  # show only first

if not nan_found:
    print("  No NaN/Inf grads found!")

print("\n--- All grad stats (first 5 params) ---")
for i, (name, p) in enumerate(model.named_parameters()):
    if p.grad is None: continue
    if i >= 5: break
    print(f"  {name}: min={p.grad.min().item():.4e} max={p.grad.max().item():.4e} mean={p.grad.mean().item():.4e}")

# Also check logits
with torch.no_grad():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out2 = model(**batch)
    print(f"\nlogits shape: {out2.logits.shape}, dtype: {out2.logits.dtype}")
    print(f"logits finite: {torch.isfinite(out2.logits).all().item()}")
    print(f"logits max: {out2.logits.abs().max().item():.4e}")

print("\nDone.")
