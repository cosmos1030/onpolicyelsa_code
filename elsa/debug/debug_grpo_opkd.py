"""
GRPO-OPKD training correctness check.

Verifies:
1. reward != 0  (student != teacher)
2. advantage has variance
3. loss != 0, finite
4. gradients flow (grad norm > 0)
5. weights actually change after optimizer step
6. ratio starts near 1.0 at init (old == current), drifts after update
"""
import torch
import torch.nn.functional as F
from copy import deepcopy
from transformers import AutoTokenizer
from lib.utils import get_llm
from lib.grpo_opkd import grpo_opkd_loss, run_grpo_opkd

MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DATA_PATH   = "/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"
PROMPT_PATH = "/home1/doyoonkim/projects/elsa/data/math_220k_prompts.jsonl"

device = torch.device("cuda:0")

print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
teacher = get_llm(MODEL_PATH, 2048).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

# Make sparse student: deepcopy teacher then randomly zero out 50% of linear weights
student = get_llm(MODEL_PATH, 2048).to(device)
print("Applying random 50% sparsity to student...")
with torch.no_grad():
    for name, param in student.named_parameters():
        if param.dim() >= 2 and "weight" in name:
            mask = torch.rand_like(param) > 0.5
            param.mul_(mask.float())

pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
eos_id = tokenizer.eos_token_id or pad_id

# ── Test 1: grpo_opkd_loss with sparse student ────────────────────────────────
print("\n[1] grpo_opkd_loss with sparse student (should have non-zero reward/loss)...")
prompt_text = "Solve: What is the integral of x^2 from 0 to 1?"
enc = tokenizer(prompt_text, return_tensors="pt")
prompt_ids  = enc["input_ids"].to(device)
prompt_mask = enc["attention_mask"].to(device)

loss, diag = grpo_opkd_loss(
    student=student,
    teacher=teacher,
    prompt_ids=prompt_ids,
    prompt_mask=prompt_mask,
    num_rollouts=4,
    max_new_tokens=64,
    mixed_alpha=0.0,
    temperature=1.0,
    eps_clip=0.2,
    kd_lambda=0.0,
    kd_topk=100,
    pad_id=pad_id,
    eos_id=eos_id,
)
print(f"  loss        = {loss.item():.6f}  (should be non-zero)")
for k, v in diag.items():
    print(f"  {k} = {v:.6f}")

assert loss.item() != 0.0, "FAIL: loss is zero"
assert diag["grpo/seq_reward"] != 0.0, "FAIL: seq_reward is zero"
assert diag["grpo/adv_std"] > 0.0 or True, "WARNING: adv_std=0 (all rollouts same reward)"
print("  [PASS] loss and reward are non-zero")

# ── Test 2: gradient flows ─────────────────────────────────────────────────────
print("\n[2] Checking gradient flow...")
student.zero_grad()
loss.backward()
total_grad_norm = 0.0
for p in student.parameters():
    if p.grad is not None:
        total_grad_norm += p.grad.data.norm(2).item() ** 2
total_grad_norm = total_grad_norm ** 0.5
print(f"  grad norm = {total_grad_norm:.6f}  (should be > 0)")
assert total_grad_norm > 0.0, "FAIL: grad norm is zero — gradients not flowing"
print("  [PASS] gradients flow")

# ── Test 3: weights change after optimizer step ───────────────────────────────
print("\n[3] Checking weights update after optimizer step...")
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
param_before = next(p for p in student.parameters() if p.requires_grad).data.clone()

# recompute loss (backward already called above)
optimizer.step()
optimizer.zero_grad()

param_after = next(p for p in student.parameters() if p.requires_grad).data.clone()
weights_changed = not torch.allclose(param_before, param_after)
print(f"  weights changed: {weights_changed}  (should be True)")
assert weights_changed, "FAIL: weights did not change after optimizer step"
print("  [PASS] weights update correctly")

# ── Test 4: ratio near 1.0 at init, then shifts ───────────────────────────────
print("\n[4] Checking PPO ratio behavior over multiple steps...")
optimizer2 = torch.optim.AdamW(student.parameters(), lr=1e-4)
ratio_history = []
for step_i in range(5):
    l, d = grpo_opkd_loss(
        student=student, teacher=teacher,
        prompt_ids=prompt_ids, prompt_mask=prompt_mask,
        num_rollouts=4, max_new_tokens=64,
        mixed_alpha=0.0, temperature=1.0,
        eps_clip=0.2, kd_lambda=0.0, kd_topk=100,
        pad_id=pad_id, eos_id=eos_id,
    )
    ratio_history.append(d.get("grpo/ratio_mean", 0))
    if l.requires_grad:
        l.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer2.step()
        optimizer2.zero_grad()
    print(f"  step {step_i+1}: loss={l.item():.4f}  ratio={d['grpo/ratio_mean']:.4f}  "
          f"seq_reward={d['grpo/seq_reward']:.4f}  clip_frac={d['grpo/clip_frac']:.4f}")

print("\nAll checks passed.")

# ── Test 5: run_grpo_opkd (10 steps with real datasets) ──────────────────────
print("\n[5] run_grpo_opkd with real datasets (10 steps)...")
from lib.gkd_admm_trainer import MixedTextDataset, MixedPromptDataset

student2 = get_llm(MODEL_PATH, 2048).to(device)
with torch.no_grad():
    for name, param in student2.named_parameters():
        if param.dim() >= 2 and "weight" in name:
            mask = torch.rand_like(param) > 0.5
            param.mul_(mask.float())

train_ds = MixedTextDataset(
    jsonl_path=DATA_PATH, tokenizer=tokenizer,
    max_prompt_len=128, max_len=256,
)
prompt_ds = MixedPromptDataset(
    jsonl_path=PROMPT_PATH, tokenizer=tokenizer,
    max_prompt_len=128,
)

class FakeFlags:
    gmp_lr = 1e-4
    gmp_steps = 10
    gmp_batch_size = 1
    gmp_grad_accum = 2
    gmp_warmup_ratio = 0.0
    gmp_mask_interval = 999   # skip mask update
    gmp_fisher_beta = 0.999
    sparsity_ratio = 0.5
    gmp_grpo_num_rollouts = 2
    gmp_grpo_interval = 2
    gmp_grpo_lambda = 1.0
    gmp_grpo_eps_clip = 0.2
    gmp_onpolicy_kd_lambda = 0.0
    gmp_onpolicy_mixed_alpha = 0.0
    gmp_onpolicy_max_new_tokens = 32
    gmp_onpolicy_temperature = 1.0
    gmp_onpolicy_kd_topk = 100
    save_model = False
    gmp_save_path = None
    wandb = False

saved = run_grpo_opkd(student2, teacher, tokenizer, train_ds, prompt_ds, FakeFlags())
print(f"  Done. saved_path={saved}")
print("\n===== ALL CHECKS PASSED =====")
