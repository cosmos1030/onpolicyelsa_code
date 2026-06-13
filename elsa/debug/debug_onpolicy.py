"""
On-policy KD debug script.
Tests: prompt load → generate → student/teacher forward → KL backward → grad_norm.
Usage: python debug_onpolicy.py [--batch_size 2] [--steps 20]
"""
import sys, math, argparse, json, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

MODEL_PATH   = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DATA_PATH    = "/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"
PROMPT_PATH  = "/home1/doyoonkim/projects/elsa/data/math_220k_prompts.jsonl"

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",       type=int,   default=2)
parser.add_argument("--steps",            type=int,   default=20)
parser.add_argument("--grad_accum",       type=int,   default=4)
parser.add_argument("--warmup",           type=int,   default=2)
parser.add_argument("--mask_interval",    type=int,   default=5)
parser.add_argument("--sparsity",         type=float, default=0.5)
parser.add_argument("--lr",              type=float, default=1e-4)
parser.add_argument("--onpolicy_interval",type=int,   default=4)
parser.add_argument("--max_new_tokens",   type=int,   default=128)
parser.add_argument("--onpolicy_lambda",  type=float, default=1.0)
parser.add_argument("--kd_topk",          type=int,   default=100)
parser.add_argument("--max_prompt_len",   type=int,   default=256)
parser.add_argument("--seq_len",          type=int,   default=512)
args = parser.parse_args()

print(f"Config: batch={args.batch_size} steps={args.steps} grad_accum={args.grad_accum} "
      f"onpolicy_interval={args.onpolicy_interval} max_new={args.max_new_tokens} lambda={args.onpolicy_lambda}")

# ── models ─────────────────────────────────────────────────────────────────
print("Loading student model...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
).cuda()
model.train()
model.config.use_cache = False

print("Loading teacher model...")
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
).cuda()
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print("Models loaded.")

# ── NTP dataset ──────────────────────────────────────────────────────────
import json
from torch.utils.data import Dataset

class NTPDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, n=200):
        self.samples = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= n: break
                text = json.loads(line).get("text", "")
                enc = tokenizer(text, max_length=max_len, truncation=True, return_tensors="pt")
                ids = enc["input_ids"][0]
                if len(ids) < 4: continue
                self.samples.append({"input_ids": ids, "attention_mask": enc["attention_mask"][0], "labels": ids.clone()})
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

_pad_tok = tok.pad_token_id

def collate_ntp(batch):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    r = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        pad = max_len - b["input_ids"].shape[0]
        r["input_ids"].append(torch.cat([b["input_ids"], torch.full((pad,), _pad_tok, dtype=torch.long)]))
        r["attention_mask"].append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        r["labels"].append(torch.cat([b["labels"], torch.full((pad,), -100, dtype=torch.long)]))
    return {k: torch.stack(v) for k, v in r.items()}

# ── Prompt dataset ────────────────────────────────────────────────────────
import sys
sys.path.insert(0, "/home1/doyoonkim/projects/elsa")
from lib.gkd_admm_trainer import MathPromptDataset, collate_prompts

ntp_ds   = NTPDataset(DATA_PATH, tok, args.seq_len)
prompt_ds = MathPromptDataset(PROMPT_PATH, tok, max_prompt_len=args.max_prompt_len, nsamples=500)

ntp_loader    = DataLoader(ntp_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_ntp)
prompt_loader = DataLoader(prompt_ds, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_prompts(tok.pad_token_id))

def inf_loader(loader):
    while True: yield from loader

ntp_iter    = inf_loader(ntp_loader)
prompt_iter = inf_loader(prompt_loader)

print(f"NTP dataset: {len(ntp_ds)}, Prompt dataset: {len(prompt_ds)}")

# ── GMP helpers ────────────────────────────────────────────────────────────
from lib.gmp_trainer import FisherAccumulator, GradualMaskManager, _find_linear_weights, _cubic_sparsity, _kl_loss

named_params = _find_linear_weights(model)
fisher       = FisherAccumulator(named_params, beta=0.999)
maskmgr      = GradualMaskManager(named_params)
optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
scheduler    = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup,
                                               num_training_steps=args.steps)

print(f"\n{'step':>5} {'ntp_loss':>10} {'op_loss':>10} {'grad_norm':>10} {'sparsity':>10}  notes")
print("-" * 65)

optimizer.zero_grad()
for step in range(1, args.steps + 1):
    notes = []

    # ── NTP micro-steps ────────────────────────────────────────────────
    accum_ntp = 0.0
    for _ in range(args.grad_accum):
        batch = {k: v.cuda() for k, v in next(ntp_iter).items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch, output_hidden_states=False)
            loss = out.loss / args.grad_accum
        if torch.isnan(loss) or torch.isinf(loss):
            notes.append("NaN_ntp"); continue
        loss.backward()
        fisher.update()
        accum_ntp += out.loss.item()

    # ── On-policy KD ──────────────────────────────────────────────────
    accum_op = float('nan')
    if step % args.onpolicy_interval == 0:
        p_batch   = next(prompt_iter)
        prompt_ids  = p_batch['input_ids'].cuda()
        prompt_mask = p_batch['attention_mask'].cuda()
        prompt_len  = int(p_batch['prompt_len'].item())

        model.config.use_cache = True
        model.eval()
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.6,
                pad_token_id=tok.pad_token_id,
            )
        model.train()
        model.config.use_cache = False
        maskmgr.apply()

        _pad_id = tok.pad_token_id
        gen_labels = generated.clone()
        gen_labels[:, :prompt_len] = -100
        gen_labels[generated == _pad_id] = -100

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            s_out = model(input_ids=generated)
            with torch.no_grad():
                t_out = teacher(input_ids=generated)
            op_kl, _ = _kl_loss(s_out.logits, t_out.logits, gen_labels, 1.0, args.kd_topk)
            op_loss = args.onpolicy_lambda * op_kl / args.grad_accum

        if torch.isnan(op_loss) or torch.isinf(op_loss):
            notes.append("NaN_op")
        else:
            op_loss.backward()
            accum_op = op_kl.item()

    # ── optimizer step ────────────────────────────────────────────────
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    if math.isnan(grad_norm) or math.isinf(grad_norm):
        notes.append("NaN_grad!")
        optimizer.zero_grad()
    else:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        maskmgr.apply()

    if step % args.mask_interval == 0:
        target_sp = _cubic_sparsity(step, args.steps, args.sparsity, args.warmup)
        if target_sp > 0:
            maskmgr.update(fisher, target_sp)

    sparsity = maskmgr.current_sparsity()
    op_str = f"{accum_op:10.4f}" if not math.isnan(accum_op) else "          -"
    note_str = " | " + ", ".join(notes) if notes else ""
    print(f"{step:>5} {accum_ntp:>10.4f} {op_str} {grad_norm:>10.4f} {sparsity:>10.4f}{note_str}")
    sys.stdout.flush()

print("\nDone.")
