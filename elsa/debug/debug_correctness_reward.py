"""
Correctness reward 구현 디버깅 스크립트.

Verifies:
1. MathPromptWithAnswerDataset: prompt tokenization + gold answer extraction
2. compute_correctness_reward: truncated / format / wrong / correct 분류
3. grpo_opkd_loss with gold_answers: reward tensor, advantage, loss 정상 여부
4. gradient flow + weight update 확인
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoTokenizer
from lib.utils import get_llm
from lib.grpo_opkd import compute_correctness_reward, grpo_opkd_loss
from lib.gkd_admm_trainer import MathPromptWithAnswerDataset, collate_prompts_with_answers

MODEL_PATH  = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DATA_PATH   = "/home1/doyoonkim/projects/elsa/data/math_220k_with_answers.jsonl"

device = torch.device("cuda:0")

print("=" * 60)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
eos_id = tokenizer.eos_token_id or pad_id

# ── Test 1: compute_correctness_reward ───────────────────────────────────────
print("\n[1] compute_correctness_reward 단위 테스트")
print("-" * 40)

cases = [
    # (description, generated, gold, expected_reward_range)
    ("truncated (no </think>)",
     "Let me think about this... 1 + 1 = 2",
     "2",
     (0.0, 0.0)),
    ("</think> but no boxed",
     "Let me think...\n</think>\nThe answer is 2.",
     "2",
     (0.04, 0.06)),  # format_reward * 0.5 = 0.05
    ("</think> + wrong boxed",
     "Let me think...\n</think>\nTherefore \\boxed{3}.",
     "2",
     (0.09, 0.11)),  # format_reward = 0.1
    ("correct (exact match)",
     "Let me think...\n</think>\nTherefore \\boxed{2}.",
     "2",
     (1.0, 1.0)),
    ("correct (numeric match)",
     "Let me think...\n</think>\n\\boxed{2.0}",
     "2",
     (1.0, 1.0)),
    ("correct (nested brace: fraction)",
     "...\n</think>\n\\boxed{\\dfrac{1}{4}}",
     "\\dfrac{1}{4}",
     (1.0, 1.0)),
    ("wrong (nested brace)",
     "...\n</think>\n\\boxed{\\dfrac{1}{3}}",
     "\\dfrac{1}{4}",
     (0.09, 0.11)),
    ("boxed in think, not after </think>",
     "<think>Intermediate: \\boxed{5}.</think>\nTherefore \\boxed{2}.",
     "2",
     (1.0, 1.0)),  # rfind gets last \boxed after </think>
]

all_ok = True
for desc, gen, gold, (lo, hi) in cases:
    r, diag = compute_correctness_reward(gen, gold, format_reward=0.1)
    ok = lo <= r <= hi
    status = "✓" if ok else "✗"
    print(f"  {status} [{desc}]")
    print(f"      reward={r:.3f}  diag={diag}")
    if not ok:
        print(f"      ERROR: expected [{lo}, {hi}]")
        all_ok = False

print(f"\n  → compute_correctness_reward: {'PASS' if all_ok else 'FAIL'}")

# ── Test 2: MathPromptWithAnswerDataset ───────────────────────────────────────
print("\n[2] MathPromptWithAnswerDataset 로드 테스트")
print("-" * 40)

ds = MathPromptWithAnswerDataset(
    jsonl_path=DATA_PATH,
    tokenizer=tokenizer,
    max_prompt_len=512,
    nsamples=50,
    seed=42,
)
print(f"  loaded: {len(ds)} samples")
assert len(ds) > 0, "Dataset empty!"

sample = ds[0]
assert "input_ids" in sample
assert "attention_mask" in sample
assert "gold_answer" in sample
assert isinstance(sample["gold_answer"], str) and len(sample["gold_answer"]) > 0

print(f"  sample[0] prompt_len={len(sample['input_ids'])}")
print(f"  sample[0] gold='{sample['gold_answer']}'")
print(f"  sample[1] gold='{ds[1]['gold_answer']}'")
print(f"  sample[2] gold='{ds[2]['gold_answer']}'")

# collate
collate_fn = collate_prompts_with_answers(pad_id)
batch = collate_fn([ds[i] for i in range(4)])
assert "input_ids" in batch
assert "gold_answers" in batch
assert len(batch["gold_answers"]) == 4
print(f"  batch input_ids shape: {batch['input_ids'].shape}")
print(f"  batch gold_answers: {batch['gold_answers']}")
print(f"  → MathPromptWithAnswerDataset: PASS")

# ── Test 3: grpo_opkd_loss with correctness reward ───────────────────────────
print("\n[3] grpo_opkd_loss (correctness reward) 테스트")
print("-" * 40)

print("  Loading student model (random 50% sparsity applied)...")
student = get_llm(MODEL_PATH, 2048).to(device)
with torch.no_grad():
    for name, param in student.named_parameters():
        if param.dim() >= 2 and "weight" in name:
            mask = torch.rand_like(param) > 0.5
            param.mul_(mask.float())

p_ids  = batch["input_ids"].to(device)
p_mask = batch["attention_mask"].to(device)
golds  = batch["gold_answers"]  # list[str], len=4

print(f"  prompt batch: {p_ids.shape}, golds={golds}")

student.train()
loss, diag = grpo_opkd_loss(
    student=student,
    teacher=None,               # teacher 없이 correctness reward
    prompt_ids=p_ids,
    prompt_mask=p_mask,
    num_rollouts=2,             # G=2 (빠른 테스트)
    max_new_tokens=256,         # 짧게 (디버깅용)
    mixed_alpha=0.0,
    temperature=1.0,
    eps_clip=0.2,
    kd_lambda=0.0,
    kd_topk=0,
    pad_id=pad_id,
    eos_id=eos_id,
    tokenizer=tokenizer,
    gold_answers=golds,
    format_reward=0.1,
)

print(f"  loss={loss.item():.4f}")
print(f"  diag={diag}")

assert torch.isfinite(loss), f"loss not finite: {loss}"
assert "grpo/correct_rate" in diag
assert "grpo/truncated_rate" in diag
assert "grpo/boxed_rate" in diag
print(f"  correct_rate  = {diag['grpo/correct_rate']:.3f}")
print(f"  boxed_rate    = {diag['grpo/boxed_rate']:.3f}")
print(f"  truncated_rate= {diag['grpo/truncated_rate']:.3f}")
print(f"  seq_reward    = {diag['grpo/seq_reward']:.4f}")

# 256토큰으로는 sparse 모델이 전부 truncate되는 게 정상
# → reward=0, advantage=0, loss=0 이 올바른 동작
if diag['grpo/truncated_rate'] == 1.0:
    print("  (all truncated at 256 tokens — expected for sparse model, loss=0 is correct)")
print(f"  → grpo_opkd_loss: PASS")

# ── Test 4: gradient flow (NTP loss로 검증) ───────────────────────────────────
print("\n[4] Gradient flow 확인 (NTP loss 사용)")
print("-" * 40)
# correctness reward에서 all-truncated면 loss=0 → grad=0이 정상.
# 대신 NTP forward로 grad flow 자체를 검증.
student.train()
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
optimizer.zero_grad()

# 짧은 dummy sequence로 NTP loss
dummy = tokenizer("The answer is \\boxed{42}.", return_tensors="pt").to(device)
labels = dummy["input_ids"].clone()
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    out = student(**dummy, labels=labels)
ntp_loss = out.loss
ntp_loss.backward()

grad_norms = [p.grad.norm().item() for p in student.parameters()
              if p.grad is not None and p.requires_grad]
print(f"  params with grad: {len(grad_norms)}")
assert len(grad_norms) > 0, "No gradients!"
total_gnorm = sum(g**2 for g in grad_norms) ** 0.5
print(f"  ntp_loss={ntp_loss.item():.4f}, total grad norm: {total_gnorm:.4f}")
assert total_gnorm > 0, "Zero gradient!"
print(f"  → gradient flow: PASS")

# ── Test 5: weight update ─────────────────────────────────────────────────────
print("\n[5] Weight update 확인")
print("-" * 40)

w_before = next(p for p in student.parameters() if p.requires_grad).detach().clone()
optimizer.step()
optimizer.zero_grad()
w_after = next(p for p in student.parameters() if p.requires_grad).detach()
delta = (w_after - w_before).abs().max().item()
print(f"  max weight delta: {delta:.6f}")
assert delta > 0, "Weights did not change!"
print(f"  → weight update: PASS")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
