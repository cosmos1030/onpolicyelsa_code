"""Probe depth sweep: the probe_vs_dense histograms, read at every depth.

Companion to plot_probe_vs_dense.py, which reads one layer (18) and says
nothing about why. This one takes the sweep npz written by
elsa/scripts/coverage_layer_sweep.py and produces two figures:

  probe_depth_hist   the SAME overlapping-histogram panels as probe_vs_dense,
                     laid out as a grid: one row per layer, one column per arm.
                     Readable the same way -- the further the two histograms
                     pull apart, the more a linear direction in the dense
                     encoder's space tells dense's continuations from this
                     model's.

  probe_depth_auc    the summary: AUC against depth, one line per arm, with the
                     same-distribution floor (dense rollouts split in half
                     within each prompt, refit at each depth) drawn underneath.
                     This is the plot that answers "which depth should I read?"

Every panel of the grid has its OWN x axis -- the probe is refit per (layer,
arm) contrast, so the dense histogram is the same data through a different lens
in every panel and is not comparable across panels. Only the overlap inside one
panel means anything. The AUC figure is the cross-panel comparison.

Everything is held out by PROMPT (grouped 5-fold); plotted scores are
out-of-fold.

What the sweep is for: at the final layer every pruned arm separates from dense
at AUC ~0.91-0.94 and the ordering between arms inverts, because last-layer
states sit next to the output distribution where "this model is pruned" is
trivially decodable. Mid-depth states still carry the distinction between arms.
Publishing the curve lets later work pick a depth deliberately instead of
inheriting ours.

Usage:
  plot_probe_depth.py --npz .../sweep_<jobid>.npz \
      [--models ours:s70,norefresh:s70,noopd55:s70] [--layers 4 9 18 27 36]
"""
import argparse
import os
import re

import numpy as np

NICE = {"ours:s70": ("SCOUT", "#0f8b7e"),
        "norefresh:s70": ("SCOUT w/o rollout refresh", "#6a51a3"),
        "noopd55:s70": ("SCOUT w/o OPD", "#54A24B"),
        "kdonly:s70": ("SCOUT w/o OPD", "#54A24B"),
        "opdonly:s70": ("OPD only", "#b08600"),
        "alps_sft:s70": ("ALPS+retrain", "#b08600")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--models", default="ours:s70,norefresh:s70,noopd55:s70")
    ap.add_argument("--reference", default="dense")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="subset for the histogram grid; default = every layer "
                         "in the npz. The AUC curve always uses all of them.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    z = np.load(a.npz)
    layers = sorted({int(m.group(1)) for m in
                     (re.match(r"L(\d+)\|", k) for k in z.files) if m})
    labs = [m for m in a.models.split(",") if m]
    outdir = a.outdir or os.path.dirname(a.npz)

    def scores(X, y, g):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, random_state=a.seed))
        s = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=g,
                              method="decision_function")
        return s, roc_auc_score(y, s)

    gD = z[f"owner_{a.reference}"]
    # Floor, refit per depth: dense rollouts split in half WITHIN each prompt.
    # A single number carried over from one layer would be wrong here -- the
    # floor is what "same distribution" looks like at that depth.
    rng = np.random.default_rng(a.seed)
    half = np.zeros(len(gD))
    for p in np.unique(gD):
        idx = np.where(gD == p)[0]
        half[rng.permutation(idx)[:len(idx) // 2]] = 1

    auc = {lab: [] for lab in labs}
    floor = []
    cache = {}
    for L in layers:
        D = z[f"L{L}|{a.reference}"].astype(np.float32)
        _, f = scores(D, half, gD)
        floor.append(f)
        for lab in labs:
            R = z[f"L{L}|{lab}"].astype(np.float32)
            gR = z[f"owner_{lab}"]
            s, v = scores(np.concatenate([D, R]),
                          np.r_[np.zeros(len(D)), np.ones(len(R))],
                          np.r_[gD, gR])
            auc[lab].append(v)
            cache[(L, lab)] = (s, len(D))
        print(f"  layer {L:>2}: floor {f:.3f} | " +
              " ".join(f"{lab.split(':')[0]} {auc[lab][-1]:.3f}" for lab in labs),
              flush=True)

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 150,
    })

    # --- figure 1: AUC vs depth -------------------------------------------
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.plot(layers, floor, color="#8a8f98", ls=":", marker="o", ms=3,
            label="same distribution (dense split in half)")
    for lab in labs:
        nice, c = NICE.get(lab, (lab, "#888"))
        ax.plot(layers, auc[lab], color=c, marker="o", ms=3.5, label=nice)
    ax.set_xlabel("encoder depth (hidden_states index, 36 = final block)")
    ax.set_ylabel("AUC vs dense rollouts")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4,
              borderpad=.2, labelspacing=.25)
    fig.tight_layout(pad=0.6)
    p1 = os.path.join(outdir, "probe_depth_auc.pdf")
    fig.savefig(p1, bbox_inches="tight")
    fig.savefig(os.path.splitext(p1)[0] + ".png", bbox_inches="tight")

    # --- figure 2: the probe_vs_dense panels, one row per depth -----------
    show = a.layers or layers
    fig, axes = plt.subplots(len(show), len(labs),
                             figsize=(3.4 * len(labs), 2.2 * len(show)),
                             squeeze=False)
    for r, L in enumerate(show):
        for c_, lab in enumerate(labs):
            ax = axes[r][c_]
            nice, col = NICE.get(lab, (lab, "#888"))
            s, nD = cache[(L, lab)]
            y = np.r_[np.zeros(nD), np.ones(len(s) - nD)]
            bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 40)
            ax.hist(s[y == 0], bins=bins, color="#3f4652", alpha=.55, lw=0,
                    label="Dense")
            ax.hist(s[y == 1], bins=bins, color=col, alpha=.62, lw=0, label=nice)
            ax.set_yticks([])
            ax.set_title(f"L{L}  {nice}   AUC {auc[lab][layers.index(L)]:.2f}",
                         pad=5)
            if r == len(show) - 1:
                ax.set_xlabel("probe score")
            if r == 0:
                ax.legend(frameon=False, loc="upper left", handlelength=1.1,
                          borderpad=.2, labelspacing=.25)
    fig.tight_layout(pad=0.6)
    p2 = os.path.join(outdir, "probe_depth_hist.pdf")
    fig.savefig(p2, bbox_inches="tight")
    fig.savefig(os.path.splitext(p2)[0] + ".png", bbox_inches="tight")

    print(f"\nwrote {p1}\nwrote {p2}")
    print("\nAUC above the same-distribution floor")
    print("  layer " + "".join(f"{NICE.get(l, (l,))[0][:14]:>16}" for l in labs))
    for i, L in enumerate(layers):
        print(f"  {L:>5} " + "".join(f"{auc[l][i] - floor[i]:>+16.3f}" for l in labs))


if __name__ == "__main__":
    main()
