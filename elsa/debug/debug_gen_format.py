"""
생성 포맷 및 토큰 길이 진단 스크립트.

모델에게 MATH 문제를 넉넉한 토큰으로 생성시켜서:
  - </think> 위치 (생성 토큰 기준)
  - \boxed{} 위치
  - truncation 여부
  - 포맷 분류 분포
를 확인한다.
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from lib.utils import get_llm
from lib.gkd_admm_trainer import MathPromptWithAnswerDataset, collate_prompts_with_answers
from lib.grpo_opkd import compute_correctness_reward

# ── 설정 ─────────────────────────────────────────────────────────────────────
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else (
    "/home1/doyoonkim/projects/elsa/models/gmp_s50pct_lr0.0001_cgrpo_0521_1252"
)
DATA_PATH     = "/home1/doyoonkim/projects/elsa/data/math_220k_with_answers.jsonl"
MAX_NEW_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
N_SAMPLES      = int(sys.argv[3]) if len(sys.argv) > 3 else 50
TEMPERATURE    = 0.6
TOP_P          = 0.95
FORMAT_REWARD  = 0.1
BATCH_SIZE     = 8

# CoT 저장 경로: logs/gen_fmt_<model_name>.jsonl
import json
_model_tag = os.path.basename(MODEL_PATH.rstrip("/"))
COT_SAVE_PATH = os.path.join(os.path.dirname(__file__), "logs", f"gen_fmt_{_model_tag}.jsonl")

device = torch.device("cuda:0")

print(f"Model : {MODEL_PATH}")
print(f"Tokens: max_new_tokens={MAX_NEW_TOKENS}, n={N_SAMPLES}")
print("=" * 60)

# ── 로드 ─────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = get_llm(MODEL_PATH, 4096).to(device)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
eos_id = tokenizer.eos_token_id or pad_id

ds = MathPromptWithAnswerDataset(
    jsonl_path=DATA_PATH,
    tokenizer=tokenizer,
    max_prompt_len=512,
    nsamples=N_SAMPLES,
    seed=42,
)
print(f"Dataset: {len(ds)} samples loaded")
print(f"CoT save: {COT_SAVE_PATH}")
_cot_f = open(COT_SAVE_PATH, "w")

collate_fn = collate_prompts_with_answers(pad_id)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ── 생성 및 분석 ──────────────────────────────────────────────────────────────
results = []
sample_idx = 0

for batch in loader:
    p_ids  = batch["input_ids"].to(device)
    p_mask = batch["attention_mask"].to(device)
    golds  = batch["gold_answers"]
    prompt_len = p_ids.shape[1]

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=p_ids,
            attention_mask=p_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

    for b in range(len(golds)):
        gold     = golds[b]
        gen_part = gen_ids[b, prompt_len:]
        gen_len  = (gen_part != pad_id).sum().item()
        gen_text = tokenizer.decode(gen_part[:gen_len], skip_special_tokens=False)

        # 포맷 분석
        has_think_close = '</think>' in gen_text
        truncated = not has_think_close

        think_close_pos = None
        pred_answer = None

        if has_think_close:
            text_before_close = gen_text[:gen_text.index('</think>')]
            think_close_pos = len(tokenizer.encode(text_before_close, add_special_tokens=False))

            answer_part = gen_text.split('</think>')[-1]
            reward, rdiag = compute_correctness_reward(gen_text, gold, FORMAT_REWARD)

            idx = answer_part.rfind(r'\boxed{')
            if idx != -1:
                depth, start = 0, idx + len(r'\boxed{')
                for j in range(start, len(answer_part)):
                    if answer_part[j] == '{': depth += 1
                    elif answer_part[j] == '}':
                        if depth == 0:
                            pred_answer = answer_part[start:j].strip()
                            break
                        depth -= 1
        else:
            reward, rdiag = 0.0, {'truncated': 1, 'has_boxed': 0, 'correct': 0}

        hit_eos = (gen_part[gen_len - 1].item() == eos_id) if gen_len > 0 else False
        sample_idx += 1

        rec = {
            'gen_len':          gen_len,
            'truncated':        truncated,
            'think_close_pos':  think_close_pos,
            'has_boxed':        rdiag['has_boxed'],
            'correct':          rdiag['correct'],
            'reward':           reward,
            'hit_eos':          hit_eos,
            'gold':             gold,
            'pred':             pred_answer,
            'gen_text':         gen_text,
        }
        results.append(rec)
        _cot_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        _cot_f.flush()

        status = ("✓CORRECT" if rdiag['correct'] else
                  ("✗wrong"   if rdiag['has_boxed'] and not truncated else
                   ("⊘nobox"  if not truncated else "…trunc")))
        think_str = f"</think>@{think_close_pos}" if think_close_pos else "NO_THINK"
        print(f"  [{sample_idx:3d}/{N_SAMPLES}] len={gen_len:5d} | {think_str:<16} | "
              f"boxed={'Y' if rdiag['has_boxed'] else 'N'} | "
              f"eos={'Y' if hit_eos else 'N'} | "
              f"{status}  gold={gold!r:.20s}  pred={str(pred_answer)!r:.20s}")

_cot_f.close()

# ── 통계 ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("통계 요약")
print("=" * 60)

n = len(results)
truncated     = [r for r in results if r['truncated']]
completed     = [r for r in results if not r['truncated']]
has_boxed     = [r for r in results if r['has_boxed']]
correct       = [r for r in results if r['correct']]

print(f"\n포맷 분포 (n={n}):")
print(f"  truncated (no </think>)        : {len(truncated):3d} / {n}  ({100*len(truncated)/n:.1f}%)")
print(f"  completed (has </think>)       : {len(completed):3d} / {n}  ({100*len(completed)/n:.1f}%)")
print(f"    └ has \\boxed{{}}             : {len(has_boxed):3d} / {n}  ({100*len(has_boxed)/n:.1f}%)")
print(f"      └ correct answer           : {len(correct):3d} / {n}  ({100*len(correct)/n:.1f}%)")

gen_lens = [r['gen_len'] for r in results]
print(f"\n생성 길이 분포 (전체, max_new_tokens={MAX_NEW_TOKENS}):")
print(f"  min={min(gen_lens)}, max={max(gen_lens)}, "
      f"mean={np.mean(gen_lens):.0f}, median={np.median(gen_lens):.0f}")
pcts = [50, 75, 90, 95, 99]
for p in pcts:
    print(f"  p{p:2d} = {np.percentile(gen_lens, p):.0f}")

if completed:
    think_poses = [r['think_close_pos'] for r in completed if r['think_close_pos']]
    print(f"\n</think> 위치 분포 (completed {len(completed)}개):")
    print(f"  min={min(think_poses)}, max={max(think_poses)}, "
          f"mean={np.mean(think_poses):.0f}, median={np.median(think_poses):.0f}")
    for p in pcts:
        print(f"  p{p:2d} = {np.percentile(think_poses, p):.0f}")

print(f"\nhit EOS: {sum(r['hit_eos'] for r in results)} / {n}")
print(f"avg reward: {np.mean([r['reward'] for r in results]):.4f}")

# 토큰 예산 추천
print("\n" + "=" * 60)
print("토큰 예산 추천")
print("=" * 60)
if completed:
    for target_pct in [50, 70, 80, 90]:
        needed = np.percentile(think_poses, target_pct) + 50  # +50 for answer after </think>
        print(f"  완성률 {target_pct}% 달성에 필요한 토큰: ~{int(needed)}")
else:
    print(f"  완성된 샘플 없음 — max_new_tokens={MAX_NEW_TOKENS}으로 모두 truncated")
    print(f"  더 큰 토큰 예산으로 재실행 필요")
