"""Token-level KL to dense, measured on fixed text and on the model's own text.

Figure 2 measures distance between mean-pooled embeddings, which is an indirect
view: averaging 1024 tokens keeps topic and style and throws away the local
choices that decide an answer. That is the likely reason the SCOUT /
SCOUT-w/o-OPD gap fades at s70 there. Token-level KL(dense || sparse) is the
direct quantity, and it can be read at the two places that matter:

  fixed   the dataset's own reasoning trace for the prompt -- text no model
          generated, standing in for the calibration data an off-policy method
          trains on
  self    the sparse model's own rollout for the SAME prompt

Only the author of the continuation changes between the two, so their
difference isolates exposure bias: a model tuned on fixed text can match dense
there and still drift once it is reading its own output.

The claim to make from this is the GAP, not the level. "SCOUT has lower
self-generated KL" invites the obvious objection that on-policy distillation
optimises approximately that. "w/o OPD matches dense on fixed text and comes
apart on its own" does not -- it is a property of off-policy training, and the
per-position curve is not something the OPD loss pins down directly.

KL(dense || sparse) is the forward KL, the direction KD trains. Positions are
the continuation only; the prompt is context, never scored.

Usage: kl_fixed_vs_selfgen.py --states_dir <pool> --out kl.json [--n_rollouts 16]
"""
import argparse
import json
import os

import numpy as np
import torch

# Overridable via --dense. The teacher must be the dense model of the SAME
# family as the checkpoints being scored -- a 4B teacher against 1.7B students
# measures the size gap, not the loss-term ablation.
_DENSE_4B = ("/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
             "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
DENSE = _DENSE_4B

# label -> checkpoint. The two SCOUT arms are the comparison; alps_sft is the
# fixed-mask recovery control, which should show the widest gap of all if the
# exposure-bias reading is right.
MODELS = {
    "ours:s50": "cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260903_142204",
    "ours:s60": "cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260903_071754",
    "ours:s70": "cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260901_080954",
    "noopd55:s50": "/home1/doyoonkim/projects/elsa/models/gmp_s50pct_lr5e-05_20260915_115020_p3476415",
    "noopd55:s60": "/home1/doyoonkim/projects/elsa/models/gmp_s60pct_lr5e-05_20260915_165740_p3530899",
    "noopd55:s70": "/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_20260915_173630_p908241",
    "alps_sft:s50": "cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260812_132642",
    "alps_sft:s60": "cosmos1030/gmp-kd3e-1-s60pct-lr1e-4_20260814_193400",
    "alps_sft:s70": "cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260814_035030",
}

CHUNK = 256          # positions per KL chunk; (chunk, 151936) fp32 is 155 MB
N_BUCKETS = 16       # position buckets for the KL-vs-depth curve


@torch.no_grad()
def kl_per_token(dense, sparse, ids, n_ctx, device):
    """KL(dense || sparse) at each continuation position of one sequence."""
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    ld = dense(x).logits[0]
    ls = sparse(x).logits[0]
    # Position i predicts token i+1, so the continuation is scored from the
    # last prompt token onward.
    lo, hi = n_ctx - 1, len(ids) - 1
    out = []
    for a in range(lo, hi, CHUNK):
        b = min(a + CHUNK, hi)
        p = torch.log_softmax(ld[a:b].float(), -1)
        q = torch.log_softmax(ls[a:b].float(), -1)
        out.append((p.exp() * (p - q)).sum(-1).cpu())
        del p, q
    del ld, ls
    return torch.cat(out).numpy().astype(np.float32) if out else np.zeros(0, np.float32)


def bucketise(vals, n=N_BUCKETS):
    """Mean KL per position bucket, so curves of different length line up."""
    if len(vals) == 0:
        return [float("nan")] * n
    edges = np.linspace(0, len(vals), n + 1).astype(int)
    return [float(vals[a:b].mean()) if b > a else float("nan")
            for a, b in zip(edges[:-1], edges[1:])]


def main():
    global DENSE, MODELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_rollouts", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=2048)
    ap.add_argument("--models", default="", help="comma-separated subset of MODELS")
    ap.add_argument("--dense", default=_DENSE_4B,
                    help="teacher; must match the students' model family")
    ap.add_argument("--model_map", default="",
                    help="label=path,label=path -- replaces MODELS entirely, "
                         "for families MODELS does not cover (e.g. 1.7B)")
    args = ap.parse_args()
    DENSE = args.dense
    if args.model_map:
        MODELS = dict(kv.split("=", 1) for kv in args.model_map.split(",") if kv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    D = args.states_dir
    pc = json.load(open(os.path.join(D, "prompts.json")))
    prompts, solutions = pc["prompts"], pc["solutions"]

    tok = AutoTokenizer.from_pretrained(DENSE, trust_remote_code=True)
    ctx = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    # The dataset's trace, cut to the same budget the rollouts were given, so
    # "fixed" and "self" score the same number of positions.
    fixed = [tok(s, add_special_tokens=False).input_ids[:args.max_new]
             for s in solutions]
    print(f"{len(prompts)} prompts | ctx {np.mean([len(c) for c in ctx]):.0f} tok "
          f"| fixed {np.mean([len(f) for f in fixed]):.0f} tok", flush=True)

    print("loading dense ...", flush=True)
    dense = AutoModelForCausalLM.from_pretrained(
        DENSE, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()

    want = [m for m in (args.models.split(",") if args.models else MODELS)
            if m in MODELS]
    res = {}
    for lab in want:
        sf = os.path.join(D, f"samples_{lab.replace(':', '_')}.json")
        if not os.path.exists(sf):
            print(f"!! no rollouts for {lab} ({sf}), skipped", flush=True)
            continue
        rolls = json.load(open(sf))
        print(f"\n=== {lab} ===", flush=True)
        sparse = AutoModelForCausalLM.from_pretrained(
            MODELS[lab], torch_dtype=torch.bfloat16,
            trust_remote_code=True).to(dev).eval()

        rec = {"fixed": [], "self": [], "fixed_curve": [], "self_curve": [],
               "fixed_n": [], "self_n": []}
        for pi in range(len(prompts)):
            c = ctx[pi]
            v = kl_per_token(dense, sparse, c + fixed[pi], len(c), dev)
            rec["fixed"].append(float(v.mean()))
            rec["fixed_curve"].append(bucketise(v))
            rec["fixed_n"].append(int(len(v)))

            per, curves, ns = [], [], []
            for r in rolls[pi][:args.n_rollouts]:
                v = kl_per_token(dense, sparse, c + list(r), len(c), dev)
                per.append(float(v.mean())); curves.append(bucketise(v))
                ns.append(int(len(v)))
            rec["self"].append(float(np.mean(per)))
            rec["self_curve"].append(np.nanmean(np.array(curves), 0).tolist())
            rec["self_n"].append(float(np.mean(ns)))
            if pi % 5 == 0:
                print(f"  p{pi}: fixed {rec['fixed'][-1]:.4f}  "
                      f"self {rec['self'][-1]:.4f}", flush=True)

        f, s = np.array(rec["fixed"]), np.array(rec["self"])
        d = s - f
        rec["summary"] = dict(
            fixed_mean=float(f.mean()), fixed_sem=float(f.std(ddof=1) / np.sqrt(len(f))),
            self_mean=float(s.mean()), self_sem=float(s.std(ddof=1) / np.sqrt(len(s))),
            gap_mean=float(d.mean()), gap_sem=float(d.std(ddof=1) / np.sqrt(len(d))),
            gap_t=float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
            n_prompts=len(f))
        print(f"  -> fixed {f.mean():.4f}  self {s.mean():.4f}  "
              f"gap {d.mean():+.4f} (t={rec['summary']['gap_t']:+.2f})", flush=True)
        res[lab] = rec

        del sparse
        import gc; gc.collect(); torch.cuda.empty_cache()
        json.dump(res, open(args.out, "w"))

    print(f"\n{'model':<16}{'fixed':>10}{'self':>10}{'gap':>10}{'t':>8}")
    for lab, rec in res.items():
        s = rec["summary"]
        print(f"{lab:<16}{s['fixed_mean']:10.4f}{s['self_mean']:10.4f}"
              f"{s['gap_mean']:+10.4f}{s['gap_t']:+8.2f}")
    json.dump(res, open(args.out, "w"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
