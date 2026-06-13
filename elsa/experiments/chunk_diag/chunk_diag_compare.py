#!/usr/bin/env python3
"""
Per-sample comparison: A (baseline) vs B (seq BoN, log p_T).
Saves full outputs for each problem so we can inspect A-correct/B-wrong cases.

Run after chunk_diag.py A and B have finished.
"""
import json, time, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

STUDENT = "/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_onpol_lmda1_20260515_234522"
TEACHER = (
    "/home1/doyoonkim/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/"
    "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
)
OUT = "/home1/doyoonkim/projects/elsa/chunk_diag_compare.json"
MAX_SAMPLES = 100
MAX_NEW_TOKENS = 8192
G = 4
TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 42


def math_score(completion, gold):
    gold_parsed = parse(f"\\boxed{{{gold}}}", extraction_mode="first_match")
    if not gold_parsed:
        gold_parsed = parse(gold, extraction_mode="first_match")
    if not gold_parsed:
        return 0.0
    answer_parsed = parse(
        completion,
        extraction_config=[LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False, malformed_operators=False, basic_latex=True,
                equations=True, boxed="all", units=True,
            ),
            boxed_match_priority=0, try_extract_without_anchor=False,
        )],
        extraction_mode="first_match",
    )
    try:
        return float(verify(gold_parsed, answer_parsed))
    except Exception:
        return 0.0


def strip_eos(tokens, eos_id, pad_id):
    pos = (tokens == eos_id).nonzero(as_tuple=True)[0]
    if len(pos) > 0:
        tokens = tokens[:pos[0] + 1]
    while tokens.numel() > 0 and tokens[-1].item() == pad_id:
        tokens = tokens[:-1]
    return tokens


@torch.no_grad()
def mean_logprob(model, input_ids, prefix_len):
    out = model(input_ids)
    logits = out.logits[0, prefix_len - 1:-1].float()
    targets = input_ids[0, prefix_len:]
    if targets.numel() == 0:
        return float("nan")
    lp = F.log_softmax(logits, dim=-1)
    return lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean().item()


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")

    print("Loading student...", flush=True)
    student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.bfloat16, device_map={"": device})
    student.eval()
    tokenizer = AutoTokenizer.from_pretrained(STUDENT)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    print("Loading teacher...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16, device_map={"": device})
    teacher.eval()

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(MAX_SAMPLES))
    problems = [ex["problem"] for ex in ds]
    golds    = [ex["answer"] for ex in ds]

    gen_kw = dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                  pad_token_id=pad_id, eos_token_id=eos_id, use_cache=True,
                  max_new_tokens=MAX_NEW_TOKENS)

    records = []
    for i, (prob, gold) in enumerate(zip(problems, golds)):
        msgs = [{"role": "user", "content": prob}]
        try:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"Problem: {prob}\n\nSolution:"

        enc = tokenizer(prompt, return_tensors="pt").to(device)
        prefix_len = enc["input_ids"].shape[1]
        prefix = enc["input_ids"][0]

        # ── Method A: single sample ──
        out_a = student.generate(**enc, **gen_kw)
        gen_a = strip_eos(out_a[0, prefix_len:], eos_id, pad_id)
        text_a = tokenizer.decode(gen_a, skip_special_tokens=True)
        correct_a = math_score(text_a, gold)
        t_score_a = mean_logprob(teacher, torch.cat([prefix, gen_a]).unsqueeze(0).to(device), prefix_len)

        # ── Method B: best-of-G by teacher logprob ──
        candidates = []
        for _ in range(G):
            out = student.generate(**enc, **gen_kw)
            gen = strip_eos(out[0, prefix_len:], eos_id, pad_id)
            text = tokenizer.decode(gen, skip_special_tokens=True)
            t_sc = mean_logprob(teacher, torch.cat([prefix, gen]).unsqueeze(0).to(device), prefix_len)
            candidates.append({"text": text, "gen_len": gen.shape[0],
                                "t_score": t_sc, "correct": math_score(text, gold)})
        candidates.sort(key=lambda x: x["t_score"], reverse=True)
        best_b = candidates[0]

        rec = {
            "idx": i,
            "gold": gold,
            "A_correct": correct_a,
            "A_len": gen_a.shape[0],
            "A_truncated": gen_a.shape[0] >= int(MAX_NEW_TOKENS * 0.99),
            "A_has_think_close": "</think>" in text_a,
            "A_has_boxed": "\\boxed{" in text_a,
            "A_t_score": t_score_a,
            "A_text": text_a[:1000],
            "B_correct": best_b["correct"],
            "B_len": best_b["gen_len"],
            "B_truncated": best_b["gen_len"] >= int(MAX_NEW_TOKENS * 0.99),
            "B_has_think_close": "</think>" in best_b["text"],
            "B_has_boxed": "\\boxed{" in best_b["text"],
            "B_best_t_score": best_b["t_score"],
            "B_all_scores": [c["t_score"] for c in candidates],
            "B_all_correct": [c["correct"] for c in candidates],
            "B_all_lens": [c["gen_len"] for c in candidates],
            "B_text": best_b["text"][:1000],
        }
        records.append(rec)

        if (i + 1) % 10 == 0:
            a_acc = sum(r["A_correct"] for r in records) / len(records)
            b_acc = sum(r["B_correct"] for r in records) / len(records)
            print(f"[{i+1}/{MAX_SAMPLES}] A={a_acc:.3f} B={b_acc:.3f}", flush=True)

    # Summary
    n = len(records)
    a_win = [r for r in records if r["A_correct"] and not r["B_correct"]]
    b_win = [r for r in records if r["B_correct"] and not r["A_correct"]]
    both  = [r for r in records if r["A_correct"] and r["B_correct"]]
    neither = [r for r in records if not r["A_correct"] and not r["B_correct"]]

    print(f"\n=== SUMMARY ===")
    print(f"A_acc={sum(r['A_correct'] for r in records)/n:.3f}  B_acc={sum(r['B_correct'] for r in records)/n:.3f}")
    print(f"A-only correct: {len(a_win)}  B-only correct: {len(b_win)}  Both: {len(both)}  Neither: {len(neither)}")
    print(f"\nA-correct, B-wrong cases ({len(a_win)}):")
    for r in a_win[:10]:
        print(f"  idx={r['idx']:3d} | A_len={r['A_len']:5d} trunc={r['A_truncated']} | "
              f"B_len={r['B_len']:5d} trunc={r['B_truncated']} | "
              f"B_t_scores={[f'{s:.2f}' for s in r['B_all_scores']]} "
              f"B_correct_per_cand={r['B_all_correct']}")

    with open(OUT, "w") as f:
        json.dump({"records": records,
                   "summary": {"a_acc": sum(r["A_correct"] for r in records)/n,
                                "b_acc": sum(r["B_correct"] for r in records)/n,
                                "a_only": len(a_win), "b_only": len(b_win),
                                "both": len(both), "neither": len(neither)}},
                  f, indent=2)
    print(f"\nSaved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
