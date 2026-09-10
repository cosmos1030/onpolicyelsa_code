"""On-policy mismatch diagnostic: does a pruned model look worse on ITS OWN
trajectory than on the dense model's?

For a dense model P and a pruned model Q, measure the SAME divergence
KL(P || Q) under two different context distributions:

    D_dense = E_{h ~ mu^P}  KL(P(.|h) || Q(.|h))     contexts from P's rollouts
    D_self  = E_{h ~ mu^Q}  KL(P(.|h) || Q(.|h))     contexts from Q's rollouts
    D_gt    = E_{h ~ dataset} KL(P(.|h) || Q(.|h))   contexts from MATH-500's
                                                     reference solutions

Only the trajectory differs -- same prompts, same models, same KL. If pruning
makes Q drift into regions where it is worse (errors compounding), D_self
exceeds D_dense, and the gap should widen with sparsity.

D_gt is the third, most off-policy condition: the reference solution is not
drawn from ANY model's distribution, so it sits further from Q's own support
than the dense rollout does. Reading the three together orders the contexts by
how off-policy they are (gt, dense, self) rather than just contrasting two.
Note it is not a like-for-like comparison with the other two: reference
solutions are human-written and much shorter than a thinking-mode rollout, so
the per-token mean is over a different kind of text, not merely a different
trajectory. Treat it as a reference point, not as a third arm of the same
experiment.

Checkpoint choice matters: use a one-shot pruning baseline (ALPS) that never
saw any on-policy signal. A method designed to reduce this very mismatch would
make the test circular, and an ALPS+recovery checkpoint has OPD mixed into it.

Both rollout sets start from the SAME prompts, so the only thing that varies
is who generated the continuation.

Usage:
  python onpolicy_mismatch_diag.py \
      --dense_model <path> --pruned_model <hf id or path> \
      --n_prompts 20 --max_new_tokens 8192 --out results.json
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F


# The ALPS calibration file (ot3_fineweb_40k_qwen3_nostrip_8192.jsonl) was
# built by scripts/build_ot3_fineweb_dataset.py as
#     load_dataset(...).shuffle(seed=42).select(range(n_ot * 3))
# then taking the first n_ot of that pool, with n_ot = 40000 * 0.8 = 32000.
# So indices 0..95999 of shuffle(seed=42) are the calibration pool. Starting
# held-out selection well past that guarantees no overlap with what the pruned
# model was calibrated on.
OT3_CALIB_POOL_END = 96_000
OT3_HELDOUT_START = 200_000


def _chat(tokenizer, user_text, enable_thinking):
    msgs = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)
    except TypeError:
        # tokenizers without the Qwen3 enable_thinking kwarg
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def build_prompts(tokenizer, n_prompts, seed, enable_thinking, source="math500"):
    """Returns (prompts, reference_continuations, ids).

    reference_continuations is the third, most off-policy context: text that
    came from neither model.

    source="ot3" is the better-matched choice. Its reference is the teacher CoT
    that OpenThoughts3 itself ships -- model-generated, <think>-prefixed, and
    thousands of tokens long, i.e. the same KIND of text as the rollouts it is
    compared against, and literally the distribution off-policy KD trains on.
    source="math500" instead uses the dataset's human-written solution, which
    averages ~530 characters and embeds Asymptote figure source; that makes it
    a different kind of text, not merely a different trajectory, so its number
    is a loose reference point rather than a comparable third arm.
    """
    from datasets import load_dataset
    import json as _json
    import random

    if source == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        idx = list(range(len(ds)))
        random.Random(seed).shuffle(idx)
        idx = idx[:n_prompts]
        prompts = [_chat(tokenizer, ds[i]["problem"], enable_thinking) for i in idx]
        return prompts, [ds[i]["solution"] for i in idx], [ds[i]["unique_id"] for i in idx]

    if source == "ot3":
        ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
        ds = ds.shuffle(seed=42)          # same seed the calibration build used
        lo = OT3_HELDOUT_START
        assert lo >= OT3_CALIB_POOL_END, "held-out window overlaps the calibration pool"
        # Materialize only the held-out window. Random access into a shuffled
        # arrow dataset walks the indices map per row and is far slower than
        # slicing the window once; 4x n_prompts leaves room for rows dropped
        # by the shape checks below.
        hi = min(lo + max(n_prompts * 4, 64), len(ds))
        ds = ds.select(range(lo, hi))
        prompts, refs, ids = [], [], []
        i = 0
        while len(prompts) < n_prompts and i < len(ds):
            conv = ds[i]["conversations"]
            if isinstance(conv, str):
                conv = _json.loads(conv)
            i += 1
            if len(conv) < 2:
                continue
            user = next((t["value"] for t in conv if t.get("from") in ("human", "user")), None)
            asst = next((t["value"] for t in conv if t.get("from") in ("gpt", "assistant")), None)
            if not user or not asst:
                continue
            prompts.append(_chat(tokenizer, user, enable_thinking))
            refs.append(asst)
            ids.append(f"ot3_shuf42_{lo + i - 1}")
        return prompts, refs, ids

    raise ValueError(f"unknown prompt source {source!r}")


def generate(model_path, prompts, max_new_tokens, temperature, gpu_mem, seed):
    """Rollouts from one model. Returns list of generated token-id lists."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=gpu_mem, max_model_len=max_new_tokens + 1024,
              enforce_eager=False, seed=seed)
    sp = SamplingParams(temperature=temperature, top_p=0.95, top_k=20,
                        max_tokens=max_new_tokens)
    outs = llm.generate(prompts, sp)
    gen = [list(o.outputs[0].token_ids) for o in outs]
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return gen


@torch.no_grad()
def kl_on_rollouts(dense, pruned, tokenizer, prompts, gens, device, chunk=512):
    """Mean KL(P || Q) over GENERATED positions only.

    Teacher-forces each (prompt + generation) through both models and averages
    the per-position KL across the generated span. Prompt positions are
    excluded: both models see identical context there, so including them would
    dilute the signal with a region the diagnostic is not about.

    The vocab-sized softmax is chunked over sequence positions to bound peak
    memory (same reason gmp_trainer._kl_loss chunks).
    """
    tot, cnt = 0.0, 0
    # Entropies come free from the log-probs the KL already needs. They separate
    # "Q tracks P better here" from "this text is easy for everyone": a gap that
    # shows up in KL *and* in H(Q) is the pruned model steering into low-entropy
    # text, not the pruned model being a better student.
    ent_q, ent_p = 0.0, 0.0
    per_pos_sum, per_pos_n = {}, {}
    for prompt, gen_ids in zip(prompts, gens):
        if not gen_ids:
            continue
        p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        ids = torch.cat([p_ids, torch.tensor(gen_ids, dtype=p_ids.dtype)]).unsqueeze(0).to(device)
        n_prompt = p_ids.numel()
        # Keep logits in bf16 and cast only the slice being scored. Casting the
        # whole (seq, ~152k-vocab) tensor to fp32 up front is 4.9GiB per model
        # at 8k tokens -- two of those plus the bf16 originals is the kind of
        # large-contiguous demand that fragments the allocator.
        lp = dense(ids).logits[0]
        lq = pruned(ids).logits[0]
        # position t's logits predict token t+1, so generated tokens are
        # predicted from positions n_prompt-1 .. len-2
        lo, hi = n_prompt - 1, ids.shape[1] - 1
        for s in range(lo, hi, chunk):
            e = min(s + chunk, hi)
            p = F.log_softmax(lp[s:e].float(), dim=-1)
            q = F.log_softmax(lq[s:e].float(), dim=-1)
            kl = F.kl_div(q, p, log_target=True, reduction="none").sum(-1)  # KL(P||Q)
            tot += float(kl.sum())
            cnt += kl.numel()
            ent_q += float((-q.exp() * q).sum(-1).sum())
            ent_p += float((-p.exp() * p).sum(-1).sum())
            for j, v in enumerate(kl.tolist()):
                b = (s - lo + j) // 256          # 256-token buckets along the generation
                per_pos_sum[b] = per_pos_sum.get(b, 0.0) + v
                per_pos_n[b] = per_pos_n.get(b, 0) + 1
        del lp, lq
        torch.cuda.empty_cache()
    curve = {str(b * 256): per_pos_sum[b] / per_pos_n[b] for b in sorted(per_pos_sum)}
    extras = {
        "H_student": ent_q / cnt if cnt else float("nan"),
        "H_teacher": ent_p / cnt if cnt else float("nan"),
    }
    return (tot / cnt if cnt else float("nan")), cnt, curve, extras


def degeneration_stats(gens, max_new_tokens, n=4):
    """How repetitive is this set of continuations?

    distinct-n is unique n-grams over total n-grams, averaged across sequences:
    1.0 means nothing repeats, and it falls toward 0 as a model loops. Reported
    alongside the truncation rate because a model that loops usually also runs
    to the token cap instead of emitting EOS.
    """
    ratios, trunc = [], 0
    for g in gens:
        if len(g) >= max_new_tokens:
            trunc += 1
        if len(g) > n:
            grams = [tuple(g[i:i + n]) for i in range(len(g) - n + 1)]
            ratios.append(len(set(grams)) / len(grams))
    return {
        f"distinct_{n}": sum(ratios) / len(ratios) if ratios else float("nan"),
        "truncation_rate": trunc / max(1, len(gens)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_model", required=True)
    ap.add_argument("--pruned_model", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--n_prompts", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--no_thinking", action="store_true")
    ap.add_argument("--prompt_source", default="math500", choices=["math500", "ot3"],
                    help="Where prompts and the reference continuation come from. "
                         "ot3 uses an OpenThoughts3 slice held out from the ALPS "
                         "calibration pool, with its own teacher CoT as reference.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.dense_model, trust_remote_code=True)
    prompts, solutions, ids = build_prompts(tok, args.n_prompts, args.seed,
                                            not args.no_thinking, args.prompt_source)
    print(f"[diag] {len(prompts)} prompts from {args.prompt_source}, "
          f"thinking={not args.no_thinking}", flush=True)

    # Rollouts first, one vLLM engine at a time (each is torn down before the
    # next), so generation never competes with the other model for memory.
    t0 = time.time()
    print("[diag] generating dense rollouts (mu^P) ...", flush=True)
    gen_dense = generate(args.dense_model, prompts, args.max_new_tokens,
                         args.temperature, args.gpu_mem, args.seed)
    t1 = time.time()
    print(f"[diag] dense rollouts done in {t1-t0:.0f}s "
          f"(mean len {sum(len(g) for g in gen_dense)/max(1,len(gen_dense)):.0f})", flush=True)

    print("[diag] generating pruned rollouts (mu^Q) ...", flush=True)
    gen_self = generate(args.pruned_model, prompts, args.max_new_tokens,
                        args.temperature, args.gpu_mem, args.seed)
    t2 = time.time()
    print(f"[diag] pruned rollouts done in {t2-t1:.0f}s "
          f"(mean len {sum(len(g) for g in gen_self)/max(1,len(gen_self)):.0f})", flush=True)

    dev = "cuda"
    dense = AutoModelForCausalLM.from_pretrained(
        args.dense_model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
    pruned = AutoModelForCausalLM.from_pretrained(
        args.pruned_model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()

    print("[diag] scoring KL on dense rollouts ...", flush=True)
    d_dense, n_d, curve_d, ex_d = kl_on_rollouts(dense, pruned, tok, prompts, gen_dense, dev)
    print("[diag] scoring KL on pruned rollouts ...", flush=True)
    d_self, n_s, curve_s, ex_s = kl_on_rollouts(dense, pruned, tok, prompts, gen_self, dev)

    # Third condition: the dataset's reference solution as the continuation.
    # Tokenized the same way a generation would be so the scoring path is
    # identical -- the only difference is where the continuation came from.
    gen_gt = [tok(sol, add_special_tokens=False).input_ids[:args.max_new_tokens]
              for sol in solutions]
    print("[diag] scoring KL on reference solutions ...", flush=True)
    d_gt, n_g, curve_g, ex_g = kl_on_rollouts(dense, pruned, tok, prompts, gen_gt, dev)

    res = {
        "label": args.label, "prompt_source": args.prompt_source,
        "dense_model": args.dense_model,
        "pruned_model": args.pruned_model, "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens, "thinking": not args.no_thinking,
        "D_dense": d_dense, "D_self": d_self, "D_gt": d_gt,
        "gap": d_self - d_dense, "gap_vs_gt": d_self - d_gt,
        "n_tokens_dense": n_d, "n_tokens_self": n_s, "n_tokens_gt": n_g,
        "mean_len_gt": sum(len(g) for g in gen_gt) / max(1, len(gen_gt)),
        "curve_gt": curve_g,
        "mean_len_dense": sum(len(g) for g in gen_dense) / max(1, len(gen_dense)),
        "mean_len_self": sum(len(g) for g in gen_self) / max(1, len(gen_self)),
        "curve_dense": curve_d, "curve_self": curve_s,
        "gen_secs_dense": t1 - t0, "gen_secs_self": t2 - t1,
        # Is a lower D_self the pruned model tracking the teacher better, or the
        # pruned model wandering into text that is easy for both? H_student and
        # distinct-4 tell those apart.
        "H_student_dense": ex_d["H_student"], "H_teacher_dense": ex_d["H_teacher"],
        "H_student_self": ex_s["H_student"], "H_teacher_self": ex_s["H_teacher"],
        "H_student_gt": ex_g["H_student"], "H_teacher_gt": ex_g["H_teacher"],
        "rep_dense": degeneration_stats(gen_dense, args.max_new_tokens),
        "rep_self": degeneration_stats(gen_self, args.max_new_tokens),
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[diag] RESULT {args.label}: D_gt={d_gt:.4f}  D_dense={d_dense:.4f}  "
          f"D_self={d_self:.4f}  |  self-dense={d_self-d_dense:+.4f}  "
          f"self-gt={d_self-d_gt:+.4f}  -> {args.out}", flush=True)
    print(f"[diag] ENTROPY H(Q) dense-rollouts={ex_d['H_student']:.4f} "
          f"self-rollouts={ex_s['H_student']:.4f}  |  "
          f"REPETITION distinct4 dense={res['rep_dense']['distinct_4']:.4f} "
          f"self={res['rep_self']['distinct_4']:.4f}  |  "
          f"trunc dense={res['rep_dense']['truncation_rate']:.2f} "
          f"self={res['rep_self']['truncation_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
