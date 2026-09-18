"""t-SNE of each model's OWN rollouts, three series only.

This is the policy-divergence figure, not the fixed-text mirror: every model
generates 64 continuations per prompt and ONE encoder (dense) reads all of
them, so what moves between the clouds is the text, not the reader.

The full version draws five methods at three sparsities and the clouds sit on
top of each other. The comparison the paper rests on is dense vs SCOUT vs
SCOUT-w/o-OPD at one sparsity, so this draws only those.

t-SNE is fit per prompt on the three clouds together -- a joint fit across
prompts would spend its resolution separating problems, which is not the
question. Distances between clusters in a t-SNE are not metric; the MMD
numbers in divergence.json are what carries a claim. This is for looking.

Usage: plot_tsne_clean_3way.py [--sparsity s70] [--layer 36] [--window late]
"""
import argparse
import os

import numpy as np

DIR = ("/home1/doyoonkim/projects/elsa/logs/policy_divergence/n30_k64_clean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--sparsity", default="s70")
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--window", default="late", choices=["early", "late"])
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    series = [
        ("dense",                      "Dense",         "#3f4652"),
        (f"ours:{a.sparsity}",         "SCOUT",         "#0f8b7e"),
        (f"noopd55:{a.sparsity}",      "SCOUT w/o OPD", "#54A24B"),
    ]
    z = np.load(os.path.join(a.dir, "pooled.npz"))
    L, W = a.layer, a.window

    fig, axes = plt.subplots(1, a.prompts, figsize=(4.2 * a.prompts, 4.6))
    axes = np.atleast_1d(axes)
    for pi in range(a.prompts):
        ax = axes[pi]
        blocks, sizes = [], []
        for key, _, _ in series:
            B = z[f"p{pi}|{key}|L{L}|{W}"].astype(np.float32)
            blocks.append(B); sizes.append(len(B))
        X = np.concatenate(blocks)
        # perplexity has to stay below the smallest cloud or t-SNE will fold
        # a 64-point cluster into its neighbours
        perp = min(a.perplexity, (min(sizes) - 1) / 3.0)
        Y = TSNE(n_components=2, perplexity=perp, init="pca",
                 random_state=a.seed).fit_transform(X)
        off = 0
        for (key, nice, c), n in zip(series, sizes):
            P = Y[off:off + n]; off += n
            ax.scatter(P[:, 0], P[:, 1], s=26, c=c, alpha=.70, linewidths=0,
                       label=nice if pi == 0 else None)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"prompt {pi}", fontsize=10)
    axes[0].legend(loc="best", fontsize=9, framealpha=.92)
    fig.suptitle(
        f"Each model's own rollouts (64 per prompt), all read by the dense "
        f"encoder -- layer {L}, {W} window, sparsity "
        f"{a.sparsity.lstrip('s')}%.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = a.out or os.path.join(a.dir, f"tsne_3way_{a.sparsity}_L{L}_{W}.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
