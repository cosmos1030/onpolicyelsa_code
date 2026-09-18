"""Replot cot_through_models for three series only.

The job's own figure draws all 13 models at once and the stars land on top of
each other. The comparison that carries the paper's claim is dense vs SCOUT vs
SCOUT-w/o-OPD at s70, so this reads the saved states and draws just those.

Depth is encoded in the marker, not by joining the points: consecutive
segments of a CoT are not neighbours in this space, so a path through them in
reading order is a set of long crossing jumps that adds ink and no
information. Segment 1 is small and pale, segment 8 large and solid.

Usage: plot_cot_clean_3way.py [--dir <cot_through_models_clean>] [--layer 36]
"""
import argparse
import json
import os

import numpy as np

SERIES = [
    ("dense",       "Dense",          "#3f4652"),
    ("ours_s70",    "SCOUT",          "#0f8b7e"),
    ("noopd55_s70", "SCOUT w/o OPD",  "#54A24B"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home1/doyoonkim/projects/elsa/logs/"
                                     "policy_divergence/cot_through_models_clean")
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    z = np.load(os.path.join(a.dir, "cot_states.npz"))
    disp = json.load(open(os.path.join(a.dir, "cot_displacement.json")))
    L = a.layer

    fig, axes = plt.subplots(1, a.prompts, figsize=(4.3 * a.prompts, 4.8))
    axes = np.atleast_1d(axes)
    for pi in range(a.prompts):
        ax = axes[pi]
        # The dense rollout cloud defines the basis, so one unit of the plot is
        # one unit of "how much dense's own generations vary".
        S = z[f"p{pi}|dense_rollouts|L{L}|scale"]
        pca = PCA(n_components=2, random_state=0).fit(S)
        Z = pca.transform(S)
        ax.scatter(Z[:, 0], Z[:, 1], s=22, alpha=.25, c="#9aa3ad", linewidths=0,
                   label="dense rollouts (scale)", zorder=1)
        for lab, nice, c in SERIES:
            B = z[f"p{pi}|{lab}|L{L}|cot"]
            Zb = pca.transform(B)
            d = disp.get(f"L{L}/{lab}")
            tag = nice if lab == "dense" else f"{nice}  (d={d[pi]:.2f})"
            n = len(Zb)
            sizes = np.linspace(40, 170, n)
            alphas = np.linspace(0.35, 1.0, n)
            for k in range(n):
                ax.scatter(Zb[k, 0], Zb[k, 1], s=sizes[k], marker="*", c=c,
                           alpha=alphas[k], edgecolors="white", linewidths=.7,
                           zorder=5)
            # one invisible proxy per series so the legend shows a solid star
            ax.scatter([], [], s=110, marker="*", c=c, edgecolors="white",
                       linewidths=.7, label=tag)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"prompt {pi}", fontsize=10)
    axes[0].legend(loc="best", fontsize=8, markerscale=.9, framealpha=.92)
    fig.suptitle(
        f"The same dataset CoT, read by each model (layer {L}). Stars are its 8 "
        f"segments -- small/pale early, large/solid late;\ngrey is dense's own "
        f"rollout cloud, the unit d is measured in. Sparsity 70%.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = a.out or os.path.join(a.dir, f"cot_3way_L{L}.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for lab, nice, _ in SERIES:
        d = disp.get(f"L{L}/{lab}")
        if d:
            print(f"  {nice:<16} mean displacement {np.mean(d):.3f}")


if __name__ == "__main__":
    main()
