#!/usr/bin/env python
"""Intro figure: how far each method's generated rollouts sit from the dense model's.

Left  -- MMD^2 to dense against sparsity, one line per method, SEM over prompts.
Right -- t-SNE of one representative prompt at 70% sparsity.

Two deliberate asymmetries between the panels:

  The dataset's CoT appears in the t-SNE as a single starred landmark but is
  absent from the MMD. OpenThoughts3 ships one trace per problem, and an MMD
  against a single point is not a distribution distance -- it would be a number
  with no sampling behaviour behind it. As a landmark it still earns its place:
  it shows the reader that the reference text sits with the dense cluster, not
  off with whatever the degraded models are doing.

  The kernel bandwidth is fixed once per prompt from the median heuristic over
  ALL models' embeddings, not per comparison. Computing it per pair gives each
  model its own kernel -- a model whose cloud is more spread gets a wider one,
  which shrinks its MMD -- so the numbers would not be on a common scale. This
  matters most for SparseGPT, whose s70 value drops from 0.919 to 0.766 when the
  bandwidth is shared.

Reads pooled.npz, so it needs no GPU and re-renders in seconds.
"""
import argparse
import json
import os

import numpy as np


def mmd2(A, B, med):
    Z = np.concatenate([A, B]).astype(np.float32)
    sq = (Z * Z).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    K = np.exp(-d2 / med)
    n = len(A)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())


def shared_bandwidth(arrays):
    Z = np.concatenate(arrays).astype(np.float32)
    sq = (Z * Z).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    return float(np.median(d2[d2 > 0]))


def embed(X, seed=0, perplexity=30):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Xp = PCA(n_components=min(50, X.shape[1], len(X) - 1),
             random_state=seed).fit_transform(X.astype(np.float32))
    p = min(perplexity, max(5, (len(Xp) - 1) // 4))
    return TSNE(n_components=2, perplexity=p, init="pca", random_state=seed,
                max_iter=1000).fit_transform(Xp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--window", default="late")
    ap.add_argument("--sparsities", nargs="+", default=["s50", "s60", "s70"])
    ap.add_argument("--tsne_sparsity", default="s70")
    ap.add_argument("--tsne_prompt", type=int, default=-1,
                    help="-1 picks the prompt whose MMD ordering is most typical")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(os.path.join(args.states_dir, "pooled.npz"))
    meta = json.load(open(os.path.join(args.states_dir, "pooled_meta.json")))
    P = meta["prompts"]
    L, w = args.layer, args.window

    # Display name -> the family key inside pooled.npz
    METHODS = [("ALPS", "alps"), ("SFT", "alps_sft"), ("SCOUT", "ours")]
    COL = {"dense": "#3f4652", "ALPS": "#d4772e", "SFT": "#b08600",
           "SCOUT": "#0f7b6c", "CoT": "#c2185b"}

    def get(pi, key):
        k = f"p{pi}|{key}|L{L}|{w}"
        return z[k].astype(np.float32) if k in z.files else None

    # ---- MMD^2, one shared bandwidth per prompt
    vals = {name: {sp: [] for sp in args.sparsities} for name, _ in METHODS}
    floor = []
    rng = np.random.default_rng(args.seed)
    for pi in range(P):
        pool = [get(pi, "dense")]
        for _, fam in METHODS:
            for sp in args.sparsities:
                v = get(pi, f"{fam}:{sp}")
                if v is not None:
                    pool.append(v)
        med = shared_bandwidth([p for p in pool if p is not None])
        D = get(pi, "dense")
        for name, fam in METHODS:
            for sp in args.sparsities:
                B = get(pi, f"{fam}:{sp}")
                if B is not None:
                    vals[name][sp].append(mmd2(D, B, med))
        # noise floor: dense against itself, split in half. Anything at this
        # level is indistinguishable from sampling noise at this sample size.
        idx = rng.permutation(len(D))
        h = len(D) // 2
        floor.append(mmd2(D[idx[:h]], D[idx[h:2 * h]], med))

    print(f"=== MMD^2 to dense (layer {L}, {w} window, n={P} prompts) ===")
    print(f"  {'':8}" + "".join(f"{sp:>18}" for sp in args.sparsities))
    for name, _ in METHODS:
        row = f"  {name:<8}"
        for sp in args.sparsities:
            v = np.array(vals[name][sp])
            row += f"{v.mean():11.4f}±{v.std(ddof=1)/np.sqrt(len(v)):.4f}"
        print(row)
    fl = np.array(floor)
    print(f"  {'floor':<8}{fl.mean():11.4f}±{fl.std(ddof=1)/np.sqrt(len(fl)):.4f}"
          f"   (dense vs dense, split-half)")

    # ---- pick the t-SNE prompt: the one whose per-method ordering matches the
    # mean ordering and whose values sit nearest the median, so the panel is
    # representative rather than the most flattering
    sp_t = args.tsne_sparsity
    if args.tsne_prompt < 0:
        means = {n: np.mean(vals[n][sp_t]) for n, _ in METHODS}
        rank = [n for n, _ in sorted(means.items(), key=lambda kv: kv[1])]
        best, best_score = 0, 1e9
        for pi in range(P):
            per = {n: vals[n][sp_t][pi] for n, _ in METHODS}
            if [n for n, _ in sorted(per.items(), key=lambda kv: kv[1])] != rank:
                continue
            score = sum(abs(per[n] - means[n]) for n in per)
            if score < best_score:
                best, best_score = pi, score
        pi_t = best
    else:
        pi_t = args.tsne_prompt
    print(f"\n  t-SNE panel: prompt {pi_t}, {sp_t}")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 9.5, "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    })
    fig = plt.figure(figsize=(7.1, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.95, 0.06, 1.0], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    axt = fig.add_subplot(gs[0, 2])

    x = np.arange(len(args.sparsities))
    for name, _ in METHODS:
        m = np.array([np.mean(vals[name][sp]) for sp in args.sparsities])
        e = np.array([np.std(vals[name][sp], ddof=1) / np.sqrt(P)
                      for sp in args.sparsities])
        ax.errorbar(x, m, yerr=e, marker="o", ms=4.5, lw=1.8, capsize=2.5,
                    color=COL[name], label=name,
                    zorder=5 if name == "SCOUT" else 3)
    ax.axhline(fl.mean(), color="#9aa3ad", lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(len(x) - 1 + 0.06, fl.mean(), "sampling floor", fontsize=7,
            color="#6d757f", va="center", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s[1:]}%" for s in args.sparsities])
    ax.set_xlabel("Sparsity")
    ax.set_ylabel("MMD$^2$ to dense rollouts")
    ax.set_xlim(-0.25, len(x) - 1 + 0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6)

    # t-SNE: clouds + the dataset CoT landmark
    Xs, labels = [], []
    for nm, key in [("dense", "dense")] + [(n, f"{f}:{sp_t}") for n, f in METHODS]:
        v = get(pi_t, key)
        if v is not None:
            Xs.append(v); labels += [nm] * len(v)
    T = get(pi_t, "teacher")
    if T is not None and len(T):
        Xs.append(T); labels += ["CoT"] * len(T)
    Z = embed(np.concatenate(Xs), args.seed)
    labels = np.array(labels)
    for nm in ["dense", "ALPS", "SFT", "SCOUT"]:
        m = labels == nm
        axt.scatter(Z[m, 0], Z[m, 1], s=7, alpha=.70, c=COL[nm], linewidths=0,
                    zorder=4 if nm == "SCOUT" else 3)
    m = labels == "CoT"
    if m.any():
        axt.scatter(Z[m, 0], Z[m, 1], s=150, marker="*", c=COL["CoT"],
                    edgecolors="white", linewidths=0.9, zorder=6,
                    label="dataset CoT")
        axt.legend(frameon=False, loc="upper left", handlelength=1.0,
                   borderpad=0.1, handletextpad=0.2)
    axt.set_xticks([]); axt.set_yticks([])
    axt.set_title(f"{sp_t[1:]}% sparsity", pad=4)
    for s in axt.spines.values():
        s.set_color("#c8cdd3")

    fig.savefig(args.out, bbox_inches="tight", dpi=400)
    fig.savefig(os.path.splitext(args.out)[0] + ".pdf", bbox_inches="tight")
    print(f"  wrote {args.out} (+ .pdf)")


if __name__ == "__main__":
    main()
