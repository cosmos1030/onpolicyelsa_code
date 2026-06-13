#!/usr/bin/env python3
"""
Chunk-wise Teacher-Verified Decoding Diagnostic

Compares 5 decoding methods on MATH-500:
  A. Student sampling (baseline)
  B. Sequence best-of-G, reward = log p_T
  C. Chunk-wise best-of-G, reward = log p_T
  D. Sequence best-of-G, reward = log p_T - log q_S
  E. Chunk-wise best-of-G, reward = log p_T - log q_S

Intended to be run as a wandb sweep with `method` in [A,B,C,D,E].
Results are logged to wandb and saved to JSON.
"""
import argparse
import json
import time
import wandb
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify
from latex2sympy2_extended import NormalizationConfig

STUDENT_PATH = "/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_onpol_lmda1_20260515_234522"
TEACHER_PATH = (
    "/home1/doyoonkim/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/"
    "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
)
OUTPUT_PATH = "/home1/doyoonkim/projects/elsa/chunk_diag_results.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--student", default=STUDENT_PATH)
    p.add_argument("--teacher", default=TEACHER_PATH)
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--chunk_size", type=int, default=128)
    p.add_argument("--num_candidates", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=OUTPUT_PATH)
    # when run standalone; sweep overrides this via wandb.config
    p.add_argument("--method", default="A",
                   help="Single method to run: A, B, C, D, or E")
    return p.parse_args()


def apply_wandb_config(args):
    """Override args with wandb.config values (set by sweep)."""
    cfg = wandb.config
    for key in vars(args):
        if hasattr(cfg, key):
            setattr(args, key, getattr(cfg, key))
    args.method = args.method.upper()


# ── scoring helpers ────────────────────────────────────────────

def _token_logprobs(model, input_ids, prefix_len):
    """Per-token log-probs for positions [prefix_len:] using model."""
    out = model(input_ids)
    logits = out.logits[0, prefix_len - 1:-1].float()  # [gen_len, vocab]
    targets = input_ids[0, prefix_len:]
    lp = F.log_softmax(logits, dim=-1)
    return lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def score_teacher(teacher, input_ids, prefix_len):
    tok = _token_logprobs(teacher, input_ids, prefix_len)
    return tok.mean().item() if tok.numel() > 0 else -1e9


@torch.no_grad()
def score_logratio(teacher, student, input_ids, prefix_len):
    t = _token_logprobs(teacher, input_ids, prefix_len)
    s = _token_logprobs(student, input_ids, prefix_len)
    return (t - s).mean().item() if t.numel() > 0 else -1e9


@torch.no_grad()
def _batch_logprobs(model, prefix_ids, chunks, prefix_len):
    """Batch forward: score each chunk given shared prefix.

    prefix_ids: [1, prefix_len]
    chunks: list of G tensors (variable length, may be empty)
    Returns: list of G mean log-prob floats.
    """
    valid = [(g, c) for g, c in enumerate(chunks) if c.numel() > 0]
    if not valid:
        return [-1e9] * len(chunks)

    max_chunk_len = max(c.shape[0] for _, c in valid)
    G_valid = len(valid)
    device = prefix_ids.device

    # Build padded batch [G_valid, prefix_len + max_chunk_len]
    rows = []
    for _, c in valid:
        pad_len = max_chunk_len - c.shape[0]
        c_pad = F.pad(c, (0, pad_len)) if pad_len > 0 else c
        rows.append(torch.cat([prefix_ids[0], c_pad]))
    batch = torch.stack(rows)  # [G_valid, prefix_len + max_chunk_len]

    out = model(batch)
    # logits[:, i] predicts token i+1  →  positions [prefix_len-1 : prefix_len+max_chunk_len-1]
    logits = out.logits[:, prefix_len - 1:-1].float()  # [G_valid, max_chunk_len, vocab]

    scores_valid = []
    for i, (_, c) in enumerate(valid):
        c_len = c.shape[0]
        lp = F.log_softmax(logits[i, :c_len], dim=-1)
        tok_lp = lp.gather(-1, c.unsqueeze(-1)).squeeze(-1)
        scores_valid.append(tok_lp.mean().item())

    # Map back to original G slots
    result = [-1e9] * len(chunks)
    for (g, _), sc in zip(valid, scores_valid):
        result[g] = sc
    return result


@torch.no_grad()
def score_chunks_batch(teacher, student, prefix_ids, chunks, prefix_len, use_logratio):
    """Score G chunks in a single teacher (+ optional student) forward pass."""
    t_scores = _batch_logprobs(teacher, prefix_ids, chunks, prefix_len)
    if not use_logratio:
        return t_scores
    s_scores = _batch_logprobs(student, prefix_ids, chunks, prefix_len)
    return [t - s for t, s in zip(t_scores, s_scores)]


# ── output helpers ─────────────────────────────────────────────

def strip_after_eos(tokens, eos_id, pad_id):
    """Remove pad tokens and trim at first eos (inclusive)."""
    eos_pos = (tokens == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0:
        tokens = tokens[:eos_pos[0] + 1]
    # strip trailing pad (shouldn't occur in batch=1 but just in case)
    while tokens.numel() > 0 and tokens[-1].item() == pad_id:
        tokens = tokens[:-1]
    return tokens


def analyze(text, gen_len, max_new_tokens):
    return {
        "has_think_close": "</think>" in text,
        "has_boxed": "\\boxed{" in text,
        "truncated": gen_len >= int(max_new_tokens * 0.99),
    }


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
            boxed_match_priority=0,
            try_extract_without_anchor=False,
        )],
        extraction_mode="first_match",
    )
    try:
        return float(verify(gold_parsed, answer_parsed))
    except Exception:
        return 0.0


def log_progress(label, i, total, results):
    if (i + 1) % 10 == 0 or i + 1 == total:
        acc = sum(r["correct"] for r in results) / len(results)
        avg_len = sum(r["gen_len"] for r in results) / len(results)
        print(f"  [{label}] [{i+1}/{total}] acc={acc:.3f} avg_len={avg_len:.0f}",
              flush=True)


def summarize(results):
    n = len(results)
    return {
        "acc": round(sum(r["correct"] for r in results) / n, 4),
        "avg_len": round(sum(r["gen_len"] for r in results) / n, 1),
        "truncation_rate": round(sum(r["truncated"] for r in results) / n, 4),
        "has_think_close_rate": round(sum(r["has_think_close"] for r in results) / n, 4),
        "boxed_rate": round(sum(r["has_boxed"] for r in results) / n, 4),
        "n": n,
    }


def gen_kwargs(args, pad_id, eos_id, max_new_tokens=None):
    return dict(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        use_cache=True,
        max_new_tokens=max_new_tokens or args.max_new_tokens,
    )


# ── Method A: vanilla student sampling ────────────────────────

def method_A(student, tokenizer, prompts, golds, args, device):
    print("[A] Student sampling baseline", flush=True)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    results = []
    for i, (prompt, gold) in enumerate(zip(prompts, golds)):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        prefix_len = enc["input_ids"].shape[1]
        out = student.generate(**enc, **gen_kwargs(args, pad_id, eos_id))
        gen = strip_after_eos(out[0, prefix_len:], eos_id, pad_id)
        text = tokenizer.decode(gen, skip_special_tokens=True)
        gen_len = gen.shape[0]
        results.append({
            "correct": math_score(text, gold),
            "gen_len": gen_len,
            **analyze(text, gen_len, args.max_new_tokens),
        })
        log_progress("A", i, len(prompts), results)
    return results


# ── Method B/D: sequence best-of-G ───────────────────────────

def method_seq_bon(student, teacher, tokenizer, prompts, golds, args, device,
                   use_logratio=False):
    label = "D" if use_logratio else "B"
    reward = "log p_T - log q_S" if use_logratio else "log p_T"
    print(f"[{label}] Sequence best-of-{args.num_candidates}, reward={reward}",
          flush=True)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    results = []
    for i, (prompt, gold) in enumerate(zip(prompts, golds)):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        prefix_len = enc["input_ids"].shape[1]
        prefix = enc["input_ids"][0]

        best_score = -1e9
        best_gen = torch.tensor([], dtype=torch.long, device=device)
        for _ in range(args.num_candidates):
            out = student.generate(**enc, **gen_kwargs(args, pad_id, eos_id))
            gen = strip_after_eos(out[0, prefix_len:], eos_id, pad_id)
            if gen.numel() == 0:
                continue
            full = torch.cat([prefix, gen]).unsqueeze(0)
            if use_logratio:
                sc = score_logratio(teacher, student, full, prefix_len)
            else:
                sc = score_teacher(teacher, full, prefix_len)
            if sc > best_score:
                best_score = sc
                best_gen = gen

        text = tokenizer.decode(best_gen, skip_special_tokens=True)
        gen_len = best_gen.shape[0]
        results.append({
            "correct": math_score(text, gold),
            "gen_len": gen_len,
            **analyze(text, gen_len, args.max_new_tokens),
        })
        log_progress(label, i, len(prompts), results)
    return results


# ── Method C/E: chunk-wise best-of-G ─────────────────────────

def method_chunk_bon(student, teacher, tokenizer, prompts, golds, args, device,
                     use_logratio=False):
    label = "E" if use_logratio else "C"
    reward = "log p_T - log q_S" if use_logratio else "log p_T"
    print(f"[{label}] Chunk best-of-{args.num_candidates}, "
          f"chunk={args.chunk_size}, reward={reward}", flush=True)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    G = args.num_candidates
    results = []

    for i, (prompt, gold) in enumerate(zip(prompts, golds)):
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        orig_prefix_len = enc["input_ids"].shape[1]
        prefix_ids = enc["input_ids"].clone()  # [1, cur_len]
        total_gen = 0
        done = False

        while total_gen < args.max_new_tokens and not done:
            this_chunk = min(args.chunk_size, args.max_new_tokens - total_gen)
            cur_prefix_len = prefix_ids.shape[1]

            # Generate G chunk candidates in a batch
            expanded = prefix_ids.expand(G, -1)
            chunk_outs = student.generate(
                input_ids=expanded,
                attention_mask=torch.ones_like(expanded),
                **gen_kwargs(args, pad_id, eos_id, max_new_tokens=this_chunk),
            )
            # chunk_outs: [G, cur_prefix_len + ≤this_chunk]

            # Extract all G chunks
            chunks = []
            chunks_has_eos = []
            for g in range(G):
                raw = chunk_outs[g, cur_prefix_len:]
                c = strip_after_eos(raw, eos_id, pad_id)
                chunks.append(c)
                chunks_has_eos.append(eos_id in raw)

            # Batch score all G chunks in one forward pass
            scores = score_chunks_batch(
                teacher, student, prefix_ids, chunks, cur_prefix_len, use_logratio
            )

            best_score = -1e9
            best_chunk = None
            best_done = False
            for g in range(G):
                if chunks[g].numel() == 0:
                    continue
                if scores[g] > best_score:
                    best_score = scores[g]
                    best_chunk = chunks[g]
                    best_done = chunks_has_eos[g]

            if best_chunk is None:
                break

            prefix_ids = torch.cat([prefix_ids, best_chunk.unsqueeze(0)], dim=1)
            total_gen += best_chunk.shape[0]
            done = best_done

        gen = prefix_ids[0, orig_prefix_len:]
        text = tokenizer.decode(gen, skip_special_tokens=True)
        gen_len = gen.shape[0]
        results.append({
            "correct": math_score(text, gold),
            "gen_len": gen_len,
            **analyze(text, gen_len, args.max_new_tokens),
        })
        log_progress(label, i, len(prompts), results)
    return results


# ── main ──────────────────────────────────────────────────────

KEY_NAMES = {
    "A": "A_student_sampling",
    "B": "B_seq_bon_teacher",
    "C": "C_chunk_bon_teacher",
    "D": "D_seq_bon_logratio",
    "E": "E_chunk_bon_logratio",
}


def main():
    args = parse_args()

    wandb.init(
        project="gmp_qwen3_1.5b_v2",
        entity="dyk6208-gwangju-institute-of-science-and-technology",
    )
    apply_wandb_config(args)

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    m = args.method

    print(f"Loading student: {args.student}", flush=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, device_map={"": device}
    )
    student.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    need_teacher = m in ["B", "C", "D", "E"]
    teacher = None
    if need_teacher:
        print(f"Loading teacher: {args.teacher}", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, dtype=torch.bfloat16, device_map={"": device}
        )
        teacher.eval()

    print(f"Loading MATH-500 ({args.max_samples} samples)", flush=True)
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.select(range(min(args.max_samples, len(ds))))
    problems = [ex["problem"] for ex in ds]
    golds = [ex["answer"] for ex in ds]

    prompts = []
    for prob in problems:
        msgs = [{"role": "user", "content": prob}]
        try:
            prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            ))
        except Exception:
            prompts.append(f"Problem: {prob}\n\nSolution:")

    print(f"Method={m}, temp={args.temperature}, top_p={args.top_p}, "
          f"max_new_tokens={args.max_new_tokens}, chunk={args.chunk_size}, G={args.num_candidates}",
          flush=True)

    method_fns = {
        "A": lambda: method_A(student, tokenizer, prompts, golds, args, device),
        "B": lambda: method_seq_bon(student, teacher, tokenizer, prompts, golds, args, device, use_logratio=False),
        "C": lambda: method_chunk_bon(student, teacher, tokenizer, prompts, golds, args, device, use_logratio=False),
        "D": lambda: method_seq_bon(student, teacher, tokenizer, prompts, golds, args, device, use_logratio=True),
        "E": lambda: method_chunk_bon(student, teacher, tokenizer, prompts, golds, args, device, use_logratio=True),
    }

    if m not in method_fns:
        raise ValueError(f"Unknown method '{m}'. Choose from A,B,C,D,E.")

    t0 = time.time()
    res = method_fns[m]()
    summary = summarize(res)
    summary["time_s"] = round(time.time() - t0, 1)
    summary["method"] = KEY_NAMES.get(m, m)

    print(f"\nResult: {summary}", flush=True)

    # log to wandb
    wandb.log({
        "diag/acc": summary["acc"],
        "diag/avg_len": summary["avg_len"],
        "diag/truncation_rate": summary["truncation_rate"],
        "diag/has_think_close_rate": summary["has_think_close_rate"],
        "diag/boxed_rate": summary["boxed_rate"],
        "diag/time_s": summary["time_s"],
        "diag/method": KEY_NAMES.get(m, m),
    })
    wandb.summary.update(summary)

    # also append to shared JSON
    try:
        import os
        existing = {}
        if os.path.exists(args.output):
            with open(args.output) as f:
                existing = json.load(f)
        existing[KEY_NAMES.get(m, m)] = summary
        with open(args.output, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"Saved to {args.output}", flush=True)
    except Exception as e:
        print(f"Warning: could not save JSON: {e}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()
