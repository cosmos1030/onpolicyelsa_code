#!/usr/bin/env python
"""Projections of the fixed-vs-self hidden states, one variant per invocation.

The first pass at this produced panels where nothing separated even where a
linear probe reached AUC 0.96. The suspicion is that the 32 states taken from
one sequence share nearly all their context, so the neighbourhood structure
t-SNE preserves is "which problem is this", not "which regime is this" -- the
regime signal is real but weak, and 2D has no room for it once problem identity
is spent.

Each variant here removes one candidate nuisance factor, so a variant that
separates tells you what was hiding the signal:

  by_prompt        colour the SAME points by prompt instead of regime. If these
                   are the clumps, the diagnosis above is confirmed.
  centered         subtract each prompt's own mean from its points, killing
                   problem identity outright, then colour by regime.
  seqmean          one point per (prompt, regime): 100 points, no within-
                   sequence correlation at all.
  nopca            t-SNE straight on 2560 dims, in case the PCA-50 step was
                   discarding the discriminative direction.
  cosine           cosine metric, since hidden-state norms grow with depth and
                   euclidean distance may be reading norm rather than direction.
  perplexity       the same embedding at perplexity 5 / 30 / 100.
  probe_proj       NOT a t-SNE. Cross-fitted projection onto the probe's own
                   direction: each point is scored by a probe trained on folds
                   that exclude its prompt, so the separation shown is
                   out-of-sample rather than the circular thing you get by
                   fitting and projecting on the same points.

probe_proj is the one that can carry a claim; the rest are for understanding
why the picture looks the way it does.
"""
import argparse
import json
import os

import numpy as np


def load(npz, meta, enc, cond, L, w):
    k = f"{enc}|{cond}|L{L}|{w}"
    if k not in meta["keys"]:
        return None, None
    return np.asarray(npz[k], np.float32), np.asarray(npz[k + "|g"], int)


def per_prompt_center(X, g):
    """Remove each prompt's own mean, so problem identity cannot organise the map."""
    Xc = X.copy()
    for p in np.unique(g):
        m = g == p
        Xc[m] -= Xc[m].mean(0, keepdims=True)
    return Xc


def seq_means(X, g):
    ps = np.unique(g)
    return np.stack([X[g == p].mean(0) for p in ps]), ps


def embed(X, seed, pca_dim=50, perplexity=30, metric="euclidean"):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    if X.shape[0] < 10:
        return None
    if pca_dim:
        d = min(pca_dim, X.shape[1], X.shape[0] - 1)
        X = PCA(n_components=d, random_state=seed).fit_transform(X)
    p = min(perplexity, max(5, (X.shape[0] - 1) // 4))
    return TSNE(n_components=2, perplexity=p, init="pca" if metric == "euclidean" else "random",
                random_state=seed, max_iter=1000, metric=metric).fit_transform(X)


def probe_scores(Xf, gf, Xs, gs, seed=0):
    """Out-of-sample decision scores, plus the AUC they imply.

    Every point is scored by a model that never saw its prompt, so the spread
    between the two classes is honest -- unlike projecting onto a direction fit
    on the same points, which separates by construction.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([Xf, Xs]).astype(np.float32)
    y = np.concatenate([np.zeros(len(Xf)), np.ones(len(Xs))])
    g = np.concatenate([gf, gs])
    n = min(5, len(set(gf.tolist())), len(set(gs.tolist())))
    if n < 2:
        return None, float("nan")
    s = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=n).split(X, y, g):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=seed))
        clf.fit(X[tr], y[tr])
        s[te] = clf.decision_function(X[te])
    return s, float(roc_auc_score(y, s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", required=True)
    ap.add_argument("--variant", required=True,
                    choices=["by_prompt", "centered", "seqmean", "nopca",
                             "cosine", "perplexity", "probe_proj"])
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--window", default="early")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    out = args.outdir or os.path.join(args.states_dir, "variants")
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(args.states_dir, "states_meta.json")))
    npz = np.load(os.path.join(args.states_dir, "states.npz"))
    labels = meta["labels"]
    L, w = args.layer, args.window
    tag = f"{args.variant}_L{L}_{w}"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_FIX, C_SELF = "#2c7fb8", "#d95f0e"
    stats = {}

    # -- perplexity sweep gets its own grid: models x perplexity
    if args.variant == "perplexity":
        perps = [5, 30, 100]
        fig, axes = plt.subplots(len(perps), len(labels),
                                 figsize=(4.0 * len(labels), 3.8 * len(perps)))
        for r, pp in enumerate(perps):
            for c, lab in enumerate(labels):
                ax = axes[r][c]
                Xf, gf = load(npz, meta, lab, "fixed", L, w)
                Xs, gs = load(npz, meta, lab, f"self:{lab}", L, w)
                if Xf is None or Xs is None:
                    ax.axis("off"); continue
                Z = embed(np.concatenate([Xf, Xs]), args.seed, 50, pp)
                ax.scatter(Z[:len(Xf), 0], Z[:len(Xf), 1], s=3, alpha=.4, c=C_FIX, linewidths=0)
                ax.scatter(Z[len(Xf):, 0], Z[len(Xf):, 1], s=3, alpha=.4, c=C_SELF, linewidths=0)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(lab)
                if c == 0:
                    ax.set_ylabel(f"perplexity {pp}")
        fig.suptitle(f"perplexity sweep — layer {L}, {w} window (blue=fixed, orange=self)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{tag}.png"), dpi=140)
        print(f"[var] wrote {tag}.png", flush=True)
        return

    fig, axes = plt.subplots(1, len(labels), figsize=(4.3 * len(labels), 4.5))
    if len(labels) == 1:
        axes = [axes]

    for ax, lab in zip(axes, labels):
        Xf, gf = load(npz, meta, lab, "fixed", L, w)
        Xs, gs = load(npz, meta, lab, f"self:{lab}", L, w)
        if Xf is None or Xs is None:
            ax.axis("off"); continue

        if args.variant == "probe_proj":
            s, auc = probe_scores(Xf, gf, Xs, gs, args.seed)
            stats[lab] = auc
            sf, ss = s[:len(Xf)], s[len(Xf):]
            bins = np.linspace(min(s.min(), -6), max(s.max(), 6), 60)
            ax.hist(sf, bins=bins, alpha=.6, color=C_FIX, label="fixed CoT", density=True)
            ax.hist(ss, bins=bins, alpha=.6, color=C_SELF, label="self rollout", density=True)
            ax.axvline(0, color="k", lw=.8, ls="--")
            ax.set_title(f"{lab}   held-out AUC {auc:.3f}")
            ax.set_yticks([])
            continue

        if args.variant == "seqmean":
            Mf, pf = seq_means(Xf, gf)
            Ms, ps = seq_means(Xs, gs)
            X, nf = np.concatenate([Mf, Ms]), len(Mf)
            _, auc = probe_scores(Mf, pf, Ms, ps, args.seed)
            stats[lab] = auc
            Z = embed(X, args.seed, pca_dim=min(30, len(X) - 1), perplexity=15)
        elif args.variant == "centered":
            Cf = per_prompt_center(Xf, gf)
            Cs = per_prompt_center(Xs, gs)
            _, auc = probe_scores(Cf, gf, Cs, gs, args.seed)
            stats[lab] = auc
            X, nf = np.concatenate([Cf, Cs]), len(Cf)
            Z = embed(X, args.seed, 50, 30)
        elif args.variant == "nopca":
            X, nf = np.concatenate([Xf, Xs]), len(Xf)
            Z = embed(X, args.seed, pca_dim=None, perplexity=30)
        elif args.variant == "cosine":
            X, nf = np.concatenate([Xf, Xs]), len(Xf)
            Z = embed(X, args.seed, 50, 30, metric="cosine")
        else:  # by_prompt
            X, nf = np.concatenate([Xf, Xs]), len(Xf)
            Z = embed(X, args.seed, 50, 30)

        if Z is None:
            ax.axis("off"); continue

        if args.variant == "by_prompt":
            g = np.concatenate([gf, gs])
            ax.scatter(Z[:, 0], Z[:, 1], s=4, alpha=.6, c=g, cmap="tab20", linewidths=0)
            ax.set_title(f"{lab}   (colour = prompt id)")
        else:
            ax.scatter(Z[:nf, 0], Z[:nf, 1], s=6 if args.variant == "seqmean" else 4,
                       alpha=.6, c=C_FIX, label="fixed CoT", linewidths=0)
            ax.scatter(Z[nf:, 0], Z[nf:, 1], s=6 if args.variant == "seqmean" else 4,
                       alpha=.6, c=C_SELF, label="self rollout", linewidths=0)
            t = lab if lab not in stats else f"{lab}   AUC {stats[lab]:.3f}"
            ax.set_title(t)
        ax.set_xticks([]); ax.set_yticks([])

    if args.variant != "by_prompt":
        axes[0].legend(loc="upper left", markerscale=3, fontsize=9, framealpha=.9)
    titles = {
        "by_prompt": "same points, coloured by PROMPT — are the clumps problems?",
        "centered": "per-prompt mean removed, then coloured by regime",
        "seqmean": "one point per (prompt, regime) — no within-sequence correlation",
        "nopca": "t-SNE on raw 2560 dims (no PCA step)",
        "cosine": "cosine metric instead of euclidean",
        "probe_proj": "cross-fitted probe score (out-of-sample, not circular)",
    }
    fig.suptitle(f"{titles[args.variant]} — layer {L}, {w} window", fontsize=12)
    fig.tight_layout()
    p = os.path.join(out, f"{tag}.png")
    fig.savefig(p, dpi=150)
    if stats:
        json.dump(stats, open(os.path.join(out, f"{tag}_auc.json"), "w"), indent=2)
        print(f"[var] {tag} AUC: " +
              "  ".join(f"{k}={v:.3f}" for k, v in stats.items()), flush=True)
    print(f"[var] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
