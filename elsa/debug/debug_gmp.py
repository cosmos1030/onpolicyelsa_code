"""
Quick GMP debug script.
Runs 60 steps (warmup=5, mask_interval=10) and prints grad_norm, loss, sparsity.
Usage: python debug_gmp.py [--batch_size 2] [--with_attention_mask]
"""
import sys
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DATA_PATH  = "/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--seq_len",    type=int, default=512)
parser.add_argument("--steps",      type=int, default=60)
parser.add_argument("--grad_accum", type=int, default=4)
parser.add_argument("--warmup",     type=int, default=5)
parser.add_argument("--mask_interval", type=int, default=10)
parser.add_argument("--sparsity",   type=float, default=0.5)
parser.add_argument("--lr",         type=float, default=1e-4)
args = parser.parse_args()

print(f"Config: batch={args.batch_size} seq={args.seq_len} steps={args.steps} "
      f"grad_accum={args.grad_accum} warmup={args.warmup} "
      f"mask_interval={args.mask_interval} sparsity={args.sparsity} lr={args.lr}")

# ── model ──────────────────────────────────────────────────────────────────
print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).cuda()
model.train()
model.config.use_cache = False
print("Model loaded.")

# ── tiny dataset from real data ────────────────────────────────────────────
import json

class TinyDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, n=200):
        self.samples = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= n: break
                text = json.loads(line).get("text", "")
                enc = tokenizer(text, max_length=max_len, truncation=True,
                                return_tensors="pt")
                ids = enc["input_ids"][0]
                if len(ids) < 4: continue
                labels = ids.clone()
                self.samples.append({"input_ids": ids, "attention_mask": enc["attention_mask"][0], "labels": labels})
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def collate(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    pad_id  = tok.pad_token_id or 0
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

print("Building dataset...")
ds = TinyDataset(DATA_PATH, tok, args.seq_len)
loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
print(f"Dataset size: {len(ds)}")

# ── GMP helpers (from gmp_trainer.py) ─────────────────────────────────────
import torch.nn as nn
from lib.gmp_trainer import (FisherAccumulator, GradualMaskManager,
                              _find_linear_weights, _cubic_sparsity)

named_params  = _find_linear_weights(model)
optimizer     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
fisher        = FisherAccumulator(named_params, optimizer)
maskmgr       = GradualMaskManager(named_params)
scheduler     = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup, num_training_steps=args.steps)

total_steps    = args.steps
warmup_steps   = args.warmup
mask_interval  = args.mask_interval
grad_accum     = args.grad_accum
final_sparsity = args.sparsity
pruning_end_steps = total_steps  # prune until the end for debug

def inf_loader(loader):
    while True: yield from loader

data_iter = inf_loader(loader)
optimizer.zero_grad()
step = 0

print(f"\n{'step':>6} {'loss':>8} {'ntp_loss':>10} {'grad_norm':>10} {'sparsity':>10}  notes")
print("-" * 65)

while step < total_steps:
    accum_loss = 0.0
    accum_ntp  = 0.0
    nan_steps  = 0

    for micro_step in range(grad_accum):
        batch = next(data_iter)
        batch = {k: v.cuda() for k, v in batch.items()}

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out      = model(**batch, output_hidden_states=False)
            ntp_loss = out.loss
            loss     = ntp_loss / grad_accum

        if torch.isnan(loss) or torch.isinf(loss):
            nan_steps += 1
            continue

        loss.backward()
        fisher.update()
        accum_loss += loss.item()
        accum_ntp  += ntp_loss.item() / grad_accum

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    notes = []
    if nan_steps:
        notes.append(f"NaN_micro={nan_steps}")

    if math.isnan(grad_norm) or math.isinf(grad_norm):
        notes.append("NaN_grad!")
        optimizer.zero_grad()
    else:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        maskmgr.apply()

    step += 1

    if step % mask_interval == 0:
        target_sp = _cubic_sparsity(step, pruning_end_steps, final_sparsity, warmup_steps)
        if target_sp > 0:
            maskmgr.update(fisher, target_sp)

    real_sp = maskmgr.current_sparsity()
    note_str = " | " + ", ".join(notes) if notes else ""
    print(f"{step:>6} {accum_loss:>8.4f} {accum_ntp:>10.4f} {grad_norm:>10.4f} {real_sp:>10.4f}{note_str}")
    sys.stdout.flush()

print("\nDone.")
