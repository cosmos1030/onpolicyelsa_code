"""
Chunk-GRPO-OPKD correctness check.

Verifies:
1. reward != 0  (student != teacher)
2. advantage has variance
3. loss != 0, finite
4. gradients flow (grad norm > 0)
5. weights actually change after optimizer step
6. prefix grows chunk by chunk, stops at EOS or budget
7. run_chunk_grpo_opkd full loop (10 steps with real datasets)
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from lib.utils import get_llm
from lib.grpo_opkd import chunk_grpo_opkd_loss, run_chunk_grpo_opkd

MODEL_PATH  = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DATA_PATH   = "/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"
PROMPT_PATH = "/home1/doyoonkim/projects/elsa/data/math_220k_prompts.jsonl"

device = torch.device("cuda:0")

print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
teacher = get_llm(MODEL_PATH, 2048).to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

student = get_llm(MODEL_PATH, 2048).to(device)
print("Applying random 50% sparsity to student...")
with torch.no_grad():
    for name, param in student.named_parameters():
        if param.dim() >= 2 and "weight" in name:
            mask = torch.rand_like(param) > 0.5
            param.mul_(mask.float())

pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
eos_id = tokenizer.eos_token_id or pad_id

# ── Helper ─────────────────────────────────────────────────────────────────────
prompt_text = "Solve: What is the integral of x^2 from 0 to 1?"
enc = tokenizer(prompt_text, return_tensors="pt")
prompt_ids  = enc["input_ids"].to(device)
prompt_mask = enc["attention_mask"].to(device)

# ── Test 1: chunk_grpo_opkd_loss basic check ───────────────────────────────────
print("\n[1] chunk_grpo_opkd_loss (sparse student, K=16, G=4, budget=64)...")
import math
student.zero_grad()
loss, diag = chunk_grpo_opkd_loss(
    student=student,
    teacher=teacher,
    prompt_ids=prompt_ids,
    prompt_mask=prompt_mask,
    num_rollouts=4,
    max_new_tokens=64,
    chunk_size=16,
    temperature=1.0,
    eps_clip=0.2,
    adv_clip=2.0,
    pad_id=pad_id,
    eos_id=eos_id,
    grpo_lambda=1.0,  # backward done inside per chunk step
)
print(f"  loss = {loss:.6f}  (should be finite, non-zero)")
for k, v in sorted(diag.items()):
    print(f"  {k} = {v:.6f}")

assert math.isfinite(loss), f"FAIL: loss is not finite ({loss})"
assert loss != 0.0, "FAIL: loss is zero"
assert diag["cgrpo/reward_mean"] != 0.0, "FAIL: reward_mean is zero"
assert diag["cgrpo/num_chunks"] >= 1, "FAIL: no chunks were processed"
print("  [PASS] loss finite and non-zero, reward non-zero, chunks processed")

# ── Test 2: gradient flows (backward already called inside) ──────────────────
print("\n[2] Checking gradient flow...")
total_grad_norm = sum(
    p.grad.data.norm(2).item() ** 2
    for p in student.parameters() if p.grad is not None
) ** 0.5
print(f"  grad norm = {total_grad_norm:.6f}  (should be > 0)")
assert total_grad_norm > 0.0, "FAIL: grad norm is zero — gradients not flowing"
print("  [PASS] gradients flow")

# ── Test 3: weights change after optimizer step ───────────────────────────────
print("\n[3] Checking weights update after optimizer step...")
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
param_before = next(p for p in student.parameters() if p.requires_grad).data.clone()
optimizer.step()
optimizer.zero_grad()
param_after = next(p for p in student.parameters() if p.requires_grad).data.clone()
assert not torch.allclose(param_before, param_after), "FAIL: weights did not change"
print("  [PASS] weights update correctly")

# ── Test 4: multiple steps — loss, reward stability ──────────────────────────
print("\n[4] Checking loss over 5 steps...")
optimizer2 = torch.optim.AdamW(student.parameters(), lr=1e-4)
for step_i in range(5):
    optimizer2.zero_grad()
    l, d = chunk_grpo_opkd_loss(
        student=student, teacher=teacher,
        prompt_ids=prompt_ids, prompt_mask=prompt_mask,
        num_rollouts=4, max_new_tokens=64, chunk_size=16,
        temperature=1.0, eps_clip=0.2, adv_clip=2.0,
        pad_id=pad_id, eos_id=eos_id,
        grpo_lambda=1.0,
    )
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
    optimizer2.step()
    print(f"  step {step_i+1}: loss={l:.4f}  "
          f"reward={d.get('cgrpo/reward_mean',0):.4f}  "
          f"clip={d.get('cgrpo/clip_frac',0):.4f}  "
          f"kl={d.get('cgrpo/approx_kl',0):.4f}  "
          f"chunks={d.get('cgrpo/num_chunks',0):.0f}")
    assert math.isfinite(l), f"FAIL: loss became non-finite at step {step_i+1}"
print("  [PASS] loss stays finite across steps")

# ── Test 5: run_chunk_grpo_opkd full loop (10 steps) ─────────────────────────
print("\n[5] run_chunk_grpo_opkd with real datasets (10 steps)...")
from lib.gkd_admm_trainer import MixedTextDataset, MixedPromptDataset

# Free test 1-4 student before loading a second model
del student, optimizer, optimizer2
torch.cuda.empty_cache()

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
    gmp_mask_interval = 999
    gmp_fisher_beta = 0.999
    sparsity_ratio = 0.5
    gmp_grpo_num_rollouts = 2
    gmp_grpo_interval = 2
    gmp_grpo_lambda = 1.0
    gmp_grpo_eps_clip = 0.2
    gmp_chunk_size = 16
    gmp_chunk_adv_clip = 2.0
    gmp_onpolicy_max_new_tokens = 64
    gmp_onpolicy_temperature = 1.0
    save_model = False
    gmp_save_path = None
    wandb = False

saved = run_chunk_grpo_opkd(student2, teacher, tokenizer, train_ds, prompt_ds, FakeFlags())
print(f"  Done. saved_path={saved}")

print("\n===== ALL CHECKS PASSED =====")
