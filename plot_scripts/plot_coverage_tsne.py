"""Does the fixed calibration CoT cover the same region as the model's own text?

This is the figure the OPD motivation needs, and it is NOT the policy-
divergence t-SNE: that one compares one model's rollouts against another's, so
both sides are generated text and neither is the dataset. Here one side is the
dataset's own reasoning trace for the prompt -- the text offline KD computes
its targets on -- and the other is what the model actually generates for that
same prompt.

Everything except the author of the continuation is held fixed: same prompts,
each continuation cut to the same span, windows read at identical absolute
token offsets, one encoder (dense) for all of it. An earlier version drew the
fixed side from random corpus documents and every model separated at AUC
0.994-0.999 including dense, which measured "reasoning text vs arbitrary web
page" -- so the matching matters.

What this figure does and does not support:
  does      fixed and on-policy text occupy different regions at all. Grouped
            5-fold CV by prompt puts a linear probe at AUC 0.948 on DENSE
            alone, so the gap is a property of "trained on written traces,
            deployed on its own", not of pruning.
  does not  any claim that pruning widens that gap or that OPD narrows it.
            Relative to dense the deltas are +0.002 (SCOUT), -0.048 (w/o OPD),
            +0.003 (ALPS+retrain) -- small, and the w/o-OPD one points the
            wrong way. coverage_postprocess.py has the intervals.

Usage: plot_coverage_tsne.py [--npz ...] [--models dense,ours:s70]
"""
import argparse
import os

import numpy as np

NPZ = ("/home1/doyoonkim/projects/elsa/logs/policy_divergence/coverage/"
       "coverage_954377.npz")
COLORS = {"dense": "#3f4652", "ours:s70": "#0f8b7e",
          "noopd55:s70": "#54A24B", "alps_sft:s70": "#b08600"}
NICE = {"dense": "Dense rollouts", "ours:s70": "SCOUT rollouts",
        "noopd55:s70": "SCOUT w/o OPD rollouts",
        "alps_sft:s70": "ALPS+retrain rollouts"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--models", default="dense,ours:s70")
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw", action="store_true",
                    help="skip the per-(prompt,depth) centring; the plot is "
                         "then dominated by reading depth and shows nothing")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    z = np.load(a.npz)
    labs = [m for m in a.models.split(",") if m]
    F, gF = z["fixed"].astype(np.float32), z["owner_fixed"]
    prompts = sorted(set(gF.tolist()))[:a.prompts]

    fig, axes = plt.subplots(1, len(prompts), figsize=(4.2 * len(prompts), 4.6))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, prompts):
        Xs, tag = [F[gF == p]], [("fixed", len(F[gF == p]))]
        for m in labs:
            R = z[m].astype(np.float32)[z[f"owner_{m}"] == p]
            Xs.append(R); tag.append((m, len(R)))
        X = np.concatenate(Xs)
        if not a.raw:
            # Within one prompt the strongest axis is reading DEPTH: window k
            # of every source sits together, so the raw plot draws eight little
            # clumps of "the same offset, three authors" and the source is
            # invisible. Both sides were read at identical offsets by
            # construction, so subtracting each window's own mean removes the
            # nuisance axis and leaves the author effect -- the same pairing
            # coverage_postprocess.py measures.
            k = len(Xs[0])              # windows per continuation
            idx = np.concatenate([np.arange(n) % k for _, n in tag])
            for w in range(k):
                m = idx == w
                if m.sum() > 1:
                    X[m] -= X[m].mean(0)
        # perplexity below the smallest cloud -- the fixed side is ~7 windows
        # per prompt and folds into a neighbour if this is left at the default
        perp = max(2.0, min(20.0, (min(n for _, n in tag) - 1) / 2.0))
        Y = TSNE(n_components=2, perplexity=perp, init="pca",
                 random_state=a.seed).fit_transform(X)
        off = 0
        for name, n in tag:
            P = Y[off:off + n]; off += n
            if name == "fixed":
                ax.scatter(P[:, 0], P[:, 1], s=210, marker="*", c="#d62728",
                           edgecolors="white", linewidths=1.0, zorder=6,
                           label="Dataset CoT (fixed KD target)" if p == prompts[0] else None)
            else:
                ax.scatter(P[:, 0], P[:, 1], s=22, c=COLORS.get(name, "#888"),
                           alpha=.62, linewidths=0, zorder=3,
                           label=NICE.get(name, name) if p == prompts[0] else None)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"prompt {p}", fontsize=10)
    axes[0].legend(loc="best", fontsize=8.5, framealpha=.92)
    fig.suptitle(
        "Same prompt, same span, same offsets, one dense encoder -- only the "
        "author of the continuation changes.\nRed stars: the dataset trace "
        "offline KD trains on. Dots: what the model generates instead.",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    out = a.out or os.path.join(os.path.dirname(a.npz), "coverage_tsne.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
