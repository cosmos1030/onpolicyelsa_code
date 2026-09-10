#!/usr/bin/env python
"""Do free-running states drift away from fixed-trace states as pruning bites?

For each model (dense, and ALPS at 50/60/70%) we collect hidden states under two
regimes on the SAME held-out prompts:

  fixed  -- teacher-forcing an external CoT (the OpenThoughts3 teacher trace)
  self   -- the model's own rollout

and ask how separable the two clouds are. t-SNE panels show it; a linear probe
AUC puts a number under the picture.

Three things this script is careful about, because each one can manufacture the
result on its own:

1. POSITION. The pruned models ramble: at 70% every rollout runs to the token
   cap while the dense model averages ~4.9k. States sampled deeper in a sequence
   differ from shallow ones for reasons that have nothing to do with pruning, so
   both regimes are sampled at the same absolute token depths, from windows that
   both are guaranteed to reach.

2. TEXT. `fixed` is the same text in every panel but `self` is not -- it comes
   from whichever model is being plotted, and at 70% that text is degenerate
   (distinct-4 collapses to 0.14). A probe could then separate the clouds purely
   from token identity, which would say nothing about representations. So every
   rollout set is ALSO encoded by the dense model: if the separation is a
   property of the text, the dense encoder finds it too, and the drift reading
   is dead. The comparison that carries the claim is
       AUC(encoder=pruned, its own rollouts)  vs  AUC(encoder=dense, same tokens)

3. LEAKAGE. Tokens from one prompt land in both probe folds unless the split is
   grouped, which inflates AUC badly. Folds are grouped by prompt.

Layers: the last layer is dominated by next-token prediction, so lexical
differences alone can light it up. A middle layer is reported alongside it --
separation that survives there is the more interesting kind.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onpolicy_mismatch_diag import build_prompts, generate  # noqa: E402


def collect_states(model, tokenizer, prompts, conts, layers, windows,
                   per_window, device, tag=""):
    """Hidden states at matched token depths inside the continuation.

    Returns {layer: {window: (X, groups)}} where X is (n, hidden) float16 and
    groups holds the prompt index each row came from, for grouped CV.
    """
    out = {L: {w: {"X": [], "g": []} for w in windows} for L in layers}
    need = max(hi for _, hi in windows.values())
    for pi, (prompt, cont) in enumerate(zip(prompts, conts)):
        if not len(cont):
            continue
        p_ids = tokenizer(prompt, return_tensors="pt",
                          add_special_tokens=False).input_ids[0]
        c_ids = torch.tensor(list(cont[:need]), dtype=p_ids.dtype)
        ids = torch.cat([p_ids, c_ids]).unsqueeze(0).to(device)
        n_prompt = p_ids.numel()
        with torch.no_grad():
            # .model skips the LM head -- we never need the 152k-wide logits,
            # and skipping them is most of the memory at these lengths.
            hs = model.model(ids, output_hidden_states=True).hidden_states
        for L in layers:
            h = hs[L][0]                      # (seq, hidden)
            for wname, (lo, hi) in windows.items():
                a, b = n_prompt + lo, n_prompt + min(hi, len(c_ids))
                if b - a < per_window:
                    continue                  # this sequence does not reach here
                idx = np.linspace(a, b - 1, per_window).astype(int)
                out[L][wname]["X"].append(h[idx].float().cpu().numpy().astype(np.float16))
                out[L][wname]["g"].append(np.full(per_window, pi))
        del hs
        torch.cuda.empty_cache()
    packed = {}
    for L in layers:
        packed[L] = {}
        for w in windows:
            xs = out[L][w]["X"]
            if xs:
                packed[L][w] = (np.concatenate(xs), np.concatenate(out[L][w]["g"]))
            else:
                packed[L][w] = (np.zeros((0, 1), np.float16), np.zeros(0, int))
    if tag:
        for w in windows:
            n = packed[layers[0]][w][0].shape[0]
            cov = len(set(packed[layers[0]][w][1].tolist()))
            print(f"[drift]   {tag} window={w}: {n} states from {cov}/{len(prompts)} prompts",
                  flush=True)
    return packed


def probe_auc(Xa, ga, Xb, gb, seed=0):
    """Grouped-CV AUC for separating two state clouds. 0.5 = indistinguishable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if Xa.shape[0] < 50 or Xb.shape[0] < 50 or Xa.shape[1] != Xb.shape[1]:
        return float("nan")
    X = np.concatenate([Xa, Xb]).astype(np.float32)
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    # Same prompt index means the same problem; offsetting cloud b keeps the two
    # regimes' rows for one prompt in the same fold, which is what we want --
    # the probe must generalise to unseen problems, not memorise them.
    g = np.concatenate([ga, gb])
    n_splits = min(5, len(set(ga.tolist())), len(set(gb.tolist())))
    if n_splits < 2:
        return float("nan")
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def tsne_xy(Xa, Xb, seed=0, pca_dim=50, perplexity=30):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X = np.concatenate([Xa, Xb]).astype(np.float32)
    if X.shape[0] < 10:
        return None
    d = min(pca_dim, X.shape[1], X.shape[0] - 1)
    X = PCA(n_components=d, random_state=seed).fit_transform(X)
    p = min(perplexity, max(5, (X.shape[0] - 1) // 4))
    Z = TSNE(n_components=2, perplexity=p, init="pca", random_state=seed,
             max_iter=1000).fit_transform(X)
    return Z[:len(Xa)], Z[len(Xa):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_model", required=True)
    ap.add_argument("--pruned_models", nargs="+", required=True,
                    help="label=path pairs, e.g. s50=cosmos1030/alps-qwen3-4b-s50pct")
    ap.add_argument("--n_prompts", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=3300)
    ap.add_argument("--per_window", type=int, default=32)
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 36])
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--prompt_source", default="ot3", choices=["math500", "ot3"])
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # Both windows sit inside what every model reaches, so no cloud is sampled
    # deeper than another. 2048-3072 is included because the KL gap at matched
    # depth grew with depth -- if drift does too, that should show here.
    windows = {"early": (0, 1024), "mid": (2048, 3072)}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.dense_model, trust_remote_code=True)
    prompts, solutions, _ = build_prompts(tok, args.n_prompts, args.seed, True,
                                          args.prompt_source)
    print(f"[drift] {len(prompts)} prompts from {args.prompt_source}", flush=True)

    models = {"dense": args.dense_model}
    for spec in args.pruned_models:
        lab, path = spec.split("=", 1)
        models[lab] = path
    labels = list(models)

    # -- rollouts, cached so a rerun of the analysis skips generation entirely
    rollouts = {}
    for lab, path in models.items():
        cache = os.path.join(args.outdir, f"rollout_{lab}.json")
        if os.path.exists(cache):
            rollouts[lab] = json.load(open(cache))
            print(f"[drift] {lab}: rollouts from cache", flush=True)
            continue
        print(f"[drift] generating rollouts: {lab}", flush=True)
        g = generate(path, prompts, args.max_new_tokens, args.temperature,
                     args.gpu_mem, args.seed)
        json.dump(g, open(cache, "w"))
        rollouts[lab] = g
        print(f"[drift] {lab}: mean len {sum(map(len, g)) / max(1, len(g)):.0f}",
              flush=True)

    fixed = [tok(s, add_special_tokens=False).input_ids for s in solutions]
    print(f"[drift] fixed CoT mean len {sum(map(len, fixed)) / max(1, len(fixed)):.0f}",
          flush=True)

    # -- encode. Each model sees the fixed trace and its own rollout; the dense
    # model additionally sees every pruned model's rollout (the text control).
    states = {}
    for enc in labels:
        print(f"[drift] encoding with {enc} ...", flush=True)
        m = AutoModelForCausalLM.from_pretrained(
            models[enc], torch_dtype=torch.bfloat16,
            trust_remote_code=True).to("cuda").eval()
        states[(enc, "fixed")] = collect_states(
            m, tok, prompts, fixed, args.layers, windows, args.per_window,
            "cuda", tag=f"{enc}/fixed")
        targets = labels if enc == "dense" else [enc]
        for src in targets:
            states[(enc, f"self:{src}")] = collect_states(
                m, tok, prompts, rollouts[src], args.layers, windows,
                args.per_window, "cuda", tag=f"{enc}/self:{src}")
        del m
        torch.cuda.empty_cache()

    # -- probe
    results = {}
    print("\n[drift] === linear-probe AUC (fixed vs self) ===", flush=True)
    for L in args.layers:
        for w in windows:
            print(f"  layer {L}, window {w}:", flush=True)
            for lab in labels:
                Xf, gf = states[(lab, "fixed")][L][w]
                Xs, gs = states[(lab, f"self:{lab}")][L][w]
                a_own = probe_auc(Xf, gf, Xs, gs, args.seed)
                results[f"L{L}/{w}/{lab}/own"] = a_own
                line = f"    {lab:<6} own-encoder AUC={a_own:.3f}"
                if lab != "dense":
                    Xdf, gdf = states[("dense", "fixed")][L][w]
                    Xds, gds = states[("dense", f"self:{lab}")][L][w]
                    a_dense = probe_auc(Xdf, gdf, Xds, gds, args.seed)
                    results[f"L{L}/{w}/{lab}/dense_encoder"] = a_dense
                    line += (f"   dense-encoder(same tokens)={a_dense:.3f}"
                             f"   delta={a_own - a_dense:+.3f}")
                print(line, flush=True)

    json.dump(results, open(os.path.join(args.outdir, "probe_auc.json"), "w"), indent=2)

    # -- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for L in args.layers:
        for w in windows:
            fig, axes = plt.subplots(1, len(labels), figsize=(4.2 * len(labels), 4.4))
            if len(labels) == 1:
                axes = [axes]
            for ax, lab in zip(axes, labels):
                Xf = states[(lab, "fixed")][L][w][0]
                Xs = states[(lab, f"self:{lab}")][L][w][0]
                z = tsne_xy(Xf, Xs, args.seed)
                if z is None:
                    ax.set_title(f"{lab}: no data")
                    continue
                Zf, Zs = z
                ax.scatter(Zf[:, 0], Zf[:, 1], s=4, alpha=.45, c="#2c7fb8",
                           label="fixed CoT", linewidths=0)
                ax.scatter(Zs[:, 0], Zs[:, 1], s=4, alpha=.45, c="#d95f0e",
                           label="self rollout", linewidths=0)
                auc = results.get(f"L{L}/{w}/{lab}/own", float("nan"))
                ax.set_title(f"{lab}   probe AUC {auc:.3f}")
                ax.set_xticks([]); ax.set_yticks([])
            axes[0].legend(loc="upper left", markerscale=3, fontsize=9, framealpha=.9)
            fig.suptitle(f"fixed CoT vs self rollout — layer {L}, tokens "
                         f"{windows[w][0]}-{windows[w][1]} of the continuation",
                         fontsize=12)
            fig.tight_layout()
            p = os.path.join(args.outdir, f"tsne_L{L}_{w}.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            print(f"[drift] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
