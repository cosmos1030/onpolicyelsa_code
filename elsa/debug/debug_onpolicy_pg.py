"""
PG loss debug script — rollout buffer (MiniLLM PPOSampler-style).
Tests: buffer collection, PPO epochs, rewards/IS weight stats, grad_norm.
Usage: python debug_onpolicy_pg.py
"""
import sys, math, json, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup

sys.path.insert(0, "/home1/doyoonkim/projects/elsa")
from lib.gmp_trainer import (
    FisherAccumulator, GradualMaskManager, RolloutBuffer,
    _find_linear_weights, _cubic_sparsity, _kl_loss, _pg_loss, _mixed_sample,
)
from lib.gkd_admm_trainer import MathPromptDataset, collate_prompts

MODEL_PATH  = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DATA_PATH   = "/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"
PROMPT_PATH = "/home1/doyoonkim/projects/elsa/data/math_220k_prompts.jsonl"

STEPS              = 40
GRAD_ACCUM         = 4
ONPOLICY_INTERVAL  = 2     # collect 1 rollout every 2 steps
MAX_NEW_TOKENS     = 64
LR                 = 1e-4
ONPOLICY_LAMBDA    = 1.0
PG_LAMBDA          = 1.0
SPARSITY           = 0.5
MIXED_ALPHA        = 0.2
ROLLOUT_BUFFER_SIZE = 16   # grad_accum(4)개/step → 4 steps마다 flush
PPO_EPOCHS         = 2

print(f"steps={STEPS} accum={GRAD_ACCUM} op_interval={ONPOLICY_INTERVAL} "
      f"max_new={MAX_NEW_TOKENS} alpha={MIXED_ALPHA} "
      f"buf={ROLLOUT_BUFFER_SIZE} ppo_epochs={PPO_EPOCHS}")

# ── models ───────────────────────────────────────────────────────────────────
torch.zeros(1).cuda()  # force CUDA init before flash_attn availability check

print("Loading models...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
tok.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
).cuda()
model.train()
model.config.use_cache = False

teacher = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
).cuda()
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print("Models loaded.")

# ── datasets ─────────────────────────────────────────────────────────────────
class NTPDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_len=512, n=200):
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

def collate_ntp(batch):
    pad = tok.pad_token_id
    max_len = max(b["input_ids"].shape[0] for b in batch)
    r = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        p = max_len - b["input_ids"].shape[0]
        r["input_ids"].append(torch.cat([b["input_ids"], torch.full((p,), pad, dtype=torch.long)]))
        r["attention_mask"].append(torch.cat([b["attention_mask"], torch.zeros(p, dtype=torch.long)]))
        r["labels"].append(torch.cat([b["labels"], torch.full((p,), -100, dtype=torch.long)]))
    return {k: torch.stack(v) for k, v in r.items()}

ntp_ds     = NTPDataset(DATA_PATH, tok)
prompt_ds  = MathPromptDataset(PROMPT_PATH, tok, max_prompt_len=256, nsamples=200)
ntp_loader = DataLoader(ntp_ds, batch_size=1, shuffle=True, collate_fn=collate_ntp)
prompt_loader = DataLoader(prompt_ds, batch_size=1, shuffle=True,
                           collate_fn=collate_prompts(tok.pad_token_id))

def inf_loader(loader):
    while True: yield from loader

ntp_iter    = inf_loader(ntp_loader)
prompt_iter = inf_loader(prompt_loader)

# ── training setup ────────────────────────────────────────────────────────────
named_params = _find_linear_weights(model)
fisher       = FisherAccumulator(named_params, beta=0.999)
maskmgr      = GradualMaskManager(named_params)
optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
scheduler    = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2,
                                               num_training_steps=STEPS)
rollout_buf  = RolloutBuffer()

_pad_id = tok.pad_token_id
_eos_id = tok.eos_token_id or _pad_id

print(f"\n{'step':>5} {'ntp':>8} {'kl':>8} {'pg':>8} {'gnorm':>8}  notes")
print("-" * 80)
optimizer.zero_grad()

for step in range(1, STEPS + 1):
    notes = []

    # ── NTP ──────────────────────────────────────────────────────────────────
    for _ in range(GRAD_ACCUM):
        batch = {k: v.cuda() for k, v in next(ntp_iter).items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss / GRAD_ACCUM
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()

    accum_kl = float('nan')
    accum_pg = float('nan')

    # ── on-policy rollout collection: batch all grad_accum prompts at once ──────
    if step % ONPOLICY_INTERVAL == 0:
        # Collect grad_accum prompts and left-pad to same length
        _p_batches   = [next(prompt_iter) for _ in range(GRAD_ACCUM)]
        _p_ids_list  = [b['input_ids'].cuda()        for b in _p_batches]
        _p_mask_list = [b['attention_mask'].cuda()   for b in _p_batches]
        _max_plen = max(p.shape[1] for p in _p_ids_list)
        _batch_ids = torch.cat([
            torch.cat([torch.full((1, _max_plen - p.shape[1]), _pad_id,
                                  dtype=torch.long, device='cuda'), p], dim=1)
            for p in _p_ids_list
        ], dim=0)  # (GRAD_ACCUM, _max_plen)
        _batch_mask = torch.cat([
            torch.cat([torch.zeros(1, _max_plen - m.shape[1],
                                   dtype=torch.long, device='cuda'), m], dim=1)
            for m in _p_mask_list
        ], dim=0)  # (GRAD_ACCUM, _max_plen)

        model.config.use_cache = True
        model.eval()
        generated = _mixed_sample(
            model, teacher, _batch_ids, _batch_mask,
            MAX_NEW_TOKENS, MIXED_ALPHA, 1.0, _pad_id, _eos_id,
        )  # (GRAD_ACCUM, _max_plen + gen_len)
        model.train()
        model.config.use_cache = False
        maskmgr.apply()

        gen_labels = generated.clone()
        gen_labels[:, :_max_plen] = -100
        gen_labels[generated == _pad_id] = -100

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                s_fwd = model(input_ids=generated)
                t_fwd = teacher(input_ids=generated)
            _gen_pos_mask = (gen_labels[:, 1:] != -100).float()
            _gids  = gen_labels[:, 1:].clamp(min=0)
            _s_lp  = F.log_softmax(s_fwd.logits[:, :-1].float(), dim=-1)
            _t_lp  = F.log_softmax(t_fwd.logits[:, :-1].float(), dim=-1)
            _s_tok = _s_lp.gather(-1, _gids.unsqueeze(-1)).squeeze(-1)
            _t_tok = _t_lp.gather(-1, _gids.unsqueeze(-1)).squeeze(-1)
            _buf_rewards = (_t_tok - _s_tok) * _gen_pos_mask
            _mix_prob = ((1 - MIXED_ALPHA) * _s_tok.exp()
                        + MIXED_ALPHA * _t_tok.exp()).clamp(min=1e-10)
            _buf_is_log_w = (_s_tok - _mix_prob.log()) * _gen_pos_mask
            for _i in range(GRAD_ACCUM):
                rollout_buf.add(generated[_i:_i+1], gen_labels[_i:_i+1],
                                _buf_rewards[_i:_i+1], _s_tok[_i:_i+1], _buf_is_log_w[_i:_i+1])
            _total_n_gen = int(_gen_pos_mask.sum().item())
            _total_r     = (_buf_rewards.sum(dim=1) / _gen_pos_mask.sum(dim=1).clamp(min=1)).mean().item()

        _w_mean = _buf_is_log_w[_gen_pos_mask.bool()].exp().mean().item() if _total_n_gen > 0 else float('nan')
        notes.append(f"[buf {len(rollout_buf)}/{ROLLOUT_BUFFER_SIZE}] gen={_total_n_gen} r={_total_r:.3f} w={_w_mean:.3f}")

    # ── NTP optimizer step ────────────────────────────────────────────────────
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    if math.isnan(grad_norm) or math.isinf(grad_norm):
        notes.append("NaN_grad!")
        optimizer.zero_grad()
    else:
        fisher.update()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        maskmgr.apply()

    # ── PPO update when buffer full (separate RL step per mini-batch) ─────────
    if len(rollout_buf) >= ROLLOUT_BUFFER_SIZE:
        _n_buf = len(rollout_buf)
        _last_kl = float('nan')
        _last_pg = float('nan')
        for _epoch in range(PPO_EPOCHS):
            for _bi in range(_n_buf):
                _gen2    = rollout_buf.generated[_bi].cuda()
                _glabels = rollout_buf.gen_labels[_bi].cuda()
                _stored  = rollout_buf.rewards[_bi].cuda()
                _s_old   = rollout_buf.old_s_logp[_bi].cuda()
                _is_lw   = rollout_buf.is_log_w[_bi].cuda()

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _s2 = model(input_ids=_gen2)
                    with torch.no_grad():
                        _t2 = teacher(input_ids=_gen2)
                    _kl2, _ = _kl_loss(_s2.logits, _t2.logits, _glabels,
                                       1.0, 100, reverse=True)
                    _pg2 = _pg_loss(_s2.logits, _t2.logits, _glabels,
                                    is_log_w=_is_lw,
                                    old_s_logp=_s_old,
                                    stored_rewards=_stored,
                                    cliprange=0.2,
                                    gamma=0.99,
                                    reward_clip=10.0)
                    _buf_loss = ONPOLICY_LAMBDA * _kl2 + PG_LAMBDA * _pg2
                if not (torch.isnan(_buf_loss) or torch.isinf(_buf_loss)):
                    _buf_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                fisher.update()
                optimizer.step()
                optimizer.zero_grad()
                maskmgr.apply()
                _last_kl = _kl2.item()
                _last_pg = _pg2.item()

        accum_kl = _last_kl
        accum_pg = _last_pg
        notes.append(f"PPO_update buf={_n_buf}x{PPO_EPOCHS}ep kl={_last_kl:.3f} pg={_last_pg:.3f}")
        rollout_buf.clear()

    kl_str = f"{accum_kl:8.4f}" if not math.isnan(accum_kl) else "       -"
    pg_str = f"{accum_pg:8.4f}" if not math.isnan(accum_pg) else "       -"
    print(f"{step:>5} {loss.item()*GRAD_ACCUM:>8.4f} {kl_str} {pg_str} {grad_norm:>8.4f}  {' | '.join(notes)}")
    sys.stdout.flush()

print("\nDone. Check: kl>=0, pg finite, gnorm finite, PPO updates at steps 8/16/24/32/40.")
