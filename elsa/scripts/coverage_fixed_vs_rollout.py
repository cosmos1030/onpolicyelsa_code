"""Do fixed calibration text and a model's own rollouts cover the same region?

The policy-divergence t-SNE compares one model's rollouts against another's --
every cloud in it is generated text. It cannot answer this question, and the
pooled npz cannot either: its `teacher` entry is the dataset's reference CoT,
and OpenThoughts3 ships one trace per problem, so that side is a single point
per prompt. One point has no coverage.

So this embeds both sides as distributions in the same space. The encoder is
the dense model throughout -- what moves between the two clouds is the text,
not the reader:

  fixed     the dataset's own reasoning trace FOR THE SAME PROMPT -- the text
            offline KD computes its targets on
  rollout   what the model generates FOR THAT SAME PROMPT -- the text on-policy
            KD computes its targets on

The prompt is held fixed and only the author of the continuation changes. A
first version of this drew the fixed side from random training-corpus documents
instead, and every model separated at AUC 0.994-0.999 including dense -- which
measured "reasoning continuation vs arbitrary web document", not coverage. Any
comparison here has to vary one thing.

One reference trace exists per problem, so the cloud comes from slicing each
continuation into fixed-width windows rather than from multiple traces: 30
prompts x N_WIN windows a side, matched in problem, text type and depth.

If the two clouds sit apart, "train on fixed text, deploy on your own text" is
a distribution shift with a picture attached, and the KL gap measured in
kl_fixed_vs_selfgen.py is its scalar version.

Sampling detail that matters: the fixed side is cut to the same token window as
the rollout side (continuation tokens 1024-2048, mean-pooled) so the comparison
is between texts of the same length read at the same depth, not between a long
document and a short generation.

Usage:
  coverage_fixed_vs_rollout.py --states_dir <pool> --train <jsonl> \
      --models ours:s70,noopd55:s70 --n_fixed 256 --out out.npz
"""
import argparse
import json
import os
import random

import numpy as np
import torch

DENSE = ("/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
         "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
WIN = 256        # tokens per window
N_WIN = 8        # windows per continuation -> 30 prompts x 8 = 240 points a side
SKIP = 256       # drop the opening: every continuation starts "<think> Okay, so"
                 # and those first tokens are near-identical across all sources


@torch.no_grad()
def windows(model, ctx_ids, cont_ids, device, layer, offsets):
    """Pooled states at GIVEN absolute continuation offsets.

    Offsets are passed in rather than derived per sequence, because the two
    sides have to be read at the same depth. A window taken at 12.5% of a
    12k-token reference trace sits at token 1500; the same fraction of a
    2k-token rollout sits at token 250, and layer-18 states move with absolute
    position, so a probe could separate the sources on depth alone.
    """
    need = max(offsets) + WIN
    if len(cont_ids) < need:
        return []
    ids = ctx_ids + list(cont_ids[:need])
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    hs = model.model(x, output_hidden_states=True).hidden_states[layer][0]
    b = len(ctx_ids)
    return [hs[b + o: b + o + WIN].float().mean(0).cpu().numpy() for o in offsets]


def prompt_bootstrap(fn, groups_a, groups_b, n=2000, seed=0):
    """Resample PROMPTS, not windows. Windows inside one trajectory are highly
    correlated, so treating 240 of them as 240 independent draws overstates
    confidence by roughly sqrt(N_WIN)."""
    rng = np.random.default_rng(seed)
    prompts = np.unique(np.r_[groups_a, groups_b])
    out = []
    for _ in range(n):
        pick = rng.choice(prompts, len(prompts), replace=True)
        v = fn(pick)
        if v is not None:
            out.append(v)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out \
        else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", required=True)
    ap.add_argument("--models", default="dense,ours:s70,noopd55:s70,alps_sft:s70")
    ap.add_argument("--n_roll", type=int, default=8,
                    help="rollouts per prompt; windows come from each of them")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dense", default=DENSE)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.dense, trust_remote_code=True)

    D = a.states_dir
    pc = json.load(open(os.path.join(D, "prompts.json")))
    prompts, solutions = pc["prompts"], pc["solutions"]
    ctx = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    # The fixed side: the dataset's own trace for each of THESE prompts.
    sol = [tok(s_, add_special_tokens=False).input_ids for s_ in solutions]
    print(f"[cov] {len(prompts)} prompts | ctx {np.mean([len(c) for c in ctx]):.0f} "
          f"| reference CoT {np.mean([len(s_) for s_ in sol]):.0f} tok", flush=True)

    print("[cov] loading dense encoder ...", flush=True)
    dense = AutoModelForCausalLM.from_pretrained(
        a.dense, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()

    # Per prompt, both sides are cut to the SAME length -- the shorter of the
    # reference trace and the shortest rollout used -- and windows are taken at
    # identical absolute offsets inside that span.
    roll_cache = {}
    for lab in [m for m in a.models.split(",") if m]:
        f = os.path.join(D, f"samples_{lab.replace(':', '_')}.json")
        if os.path.exists(f):
            roll_cache[lab] = json.load(open(f))
    span = []
    rng0 = random.Random(a.seed)
    picks = {}
    for lab, rolls in roll_cache.items():
        picks[lab] = []
        for pi in range(len(prompts)):
            pr = rolls[pi][:]; rng0.shuffle(pr); picks[lab].append(pr[:a.n_roll])
    for pi in range(len(prompts)):
        lens = [len(sol[pi])] + [len(r) for lab in picks for r in picks[lab][pi]]
        span.append(min(lens))
    OFF = {pi: [SKIP + i * WIN for i in range(N_WIN)
                if SKIP + (i + 1) * WIN <= span[pi]] for pi in range(len(prompts))}
    usable = [pi for pi in OFF if OFF[pi]]
    print(f"[cov] matched span per prompt: median {int(np.median(span))} tok, "
          f"{len(usable)}/{len(prompts)} prompts usable, "
          f"{int(np.median([len(OFF[pi]) for pi in usable]))} windows each", flush=True)

    blob, owner = {}, {}
    v, who = [], []
    for pi in usable:
        w = windows(dense, ctx[pi], sol[pi], dev, a.layer, OFF[pi])
        v += w; who += [pi] * len(w)
    blob["fixed"] = np.stack(v).astype(np.float16)
    owner["fixed"] = np.array(who)
    print(f"[cov] fixed: {blob['fixed'].shape} "
          f"({len(set(who))} prompts contributed)", flush=True)

    for lab in roll_cache:
        v, who = [], []
        for pi in usable:
            for r in picks[lab][pi]:
                w = windows(dense, ctx[pi], list(r), dev, a.layer, OFF[pi])
                v += w; who += [pi] * len(w)
        blob[lab] = np.stack(v).astype(np.float16)
        owner[lab] = np.array(who)
        print(f"[cov] {lab}: {blob[lab].shape}", flush=True)

    np.savez_compressed(a.out, **blob,
                        **{f"owner_{k}": v_ for k, v_ in owner.items()})
    print(f"[cov] wrote {a.out}", flush=True)

    # --- separability: grouped CV, prompt-level CI, prompt-level null ------
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def auc_of(X, y, g, k=5):
        if len(np.unique(y)) < 2 or len(np.unique(g)) < k:
            return None
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=0))
        # Grouped by prompt: all windows of one problem stay on one side of the
        # split, so the probe cannot win by memorising that problem's wording.
        return float(cross_val_score(clf, X, y, cv=GroupKFold(k), groups=g,
                                     scoring="roc_auc").mean())

    F, gF = blob["fixed"].astype(np.float32), owner["fixed"]
    print()
    for lab in blob:
        if lab == "fixed":
            continue
        R, gR = blob[lab].astype(np.float32), owner[lab]
        X = np.concatenate([F, R]); y = np.r_[np.zeros(len(F)), np.ones(len(R))]
        g = np.r_[gF, gR]
        auc = auc_of(X, y, g)

        def on(pick):
            m = np.concatenate([np.where(g == p)[0] for p in pick])
            return auc_of(X[m], y[m], g[m])
        lo, hi = prompt_bootstrap(on, gF, gR, n=200, seed=0)

        # Null: swap the source label within each prompt, keeping the counts.
        # If the two sources are exchangeable given the problem, this is the
        # distribution the observed AUC should sit inside.
        rng = np.random.default_rng(0)
        null = []
        for _ in range(200):
            yp = y.copy()
            for p in np.unique(g):
                m = np.where(g == p)[0]
                yp[m] = rng.permutation(y[m])
            v = auc_of(X, yp, g)
            if v is not None:
                null.append(v)
        pv = (1 + sum(v >= auc for v in null)) / (1 + len(null)) if null else float("nan")
        print(f"[cov] fixed vs {lab:<16} AUC={auc:.3f}  "
              f"95% CI over prompts [{lo:.3f}, {hi:.3f}]  "
              f"null mean {np.mean(null):.3f}  p={pv:.3f}", flush=True)

    print("\nBoth sides answer the SAME 30 problems, are cut to the same span,")
    print("and are read at identical token offsets -- only the author of the")
    print("continuation differs. CI and p are over PROMPTS (n=30), not windows.")


if __name__ == "__main__":
    main()
