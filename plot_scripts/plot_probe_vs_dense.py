"""How far has each sparse model's OWN text moved from dense's own text?

The companion figure (fixed CoT vs rollouts) asks whether the calibration data
covers what a model generates. This one changes the reference: both sides are
now on-policy text, dense's against the sparse model's, for the same prompts at
the same reading depths, read by the same encoder. It is the policy-divergence
question that divergence.json answers with MMD, asked with a linear probe so
the answer has a picture.

A probe score is read as: the further the two histograms pull apart, the more a
linear direction in the dense encoder's space tells dense's continuations from
this model's. AUC 0.5 would mean the model's on-policy distribution is
indistinguishable from dense's.

Each panel has its OWN x axis. The probe is refit per contrast, so the dense
points are projected onto a different direction in every panel -- the dense
histogram is the same data drawn through a different lens and is not
comparable across panels. Only the overlap inside one panel means anything,
which is why the axis is labelled with the contrast rather than "probe score".

The scale is set by a control: splitting dense's own rollouts in half within
each prompt and running the same probe gives AUC 0.511, so 0.5 really is the
floor here and the excess over it is the quantity to read.

Everything is held out by PROMPT (grouped 5-fold), so the probe cannot win by
memorising a problem's wording, and the plotted score is out-of-fold.

Usage: plot_probe_vs_dense.py [--models ours:s70,noopd55:s70]
"""
import argparse
import os

import numpy as np

NPZ = ("/home1/doyoonkim/projects/elsa/logs/policy_divergence/coverage/"
       "coverage_954377.npz")
NICE = {"ours:s70": ("SCOUT", "#0f8b7e"),
        "noopd55:s70": ("SCOUT w/o OPD", "#54A24B"),
        "alps_sft:s70": ("ALPS+retrain", "#b08600")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--models", default="ours:s70,noopd55:s70")
    ap.add_argument("--reference", default="dense")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--floor", type=float, default=0.511,
                    help="AUC of dense's own rollouts split in half, the "
                         "empirical 'same distribution' value for this probe")
    ap.add_argument("--out", default=None)
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
    D = z[a.reference].astype(np.float32)
    gD = z[f"owner_{a.reference}"]
    labs = [m for m in a.models.split(",") if m]

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 150,
    })
    fig, axes = plt.subplots(1, len(labs), figsize=(3.4 * len(labs), 2.5))
    axes = np.atleast_1d(axes)
    out = {}
    for ax, lab in zip(axes, labs):
        nice, c = NICE.get(lab, (lab, "#888"))
        R = z[lab].astype(np.float32)
        gR = z[f"owner_{lab}"]
        # Both sides are 8 rollouts x 7 windows per prompt, so the classes are
        # already balanced and no subsampling is needed.
        X = np.concatenate([D, R])
        y = np.r_[np.zeros(len(D)), np.ones(len(R))]
        g = np.r_[gD, gR]
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000,
                                               random_state=a.seed))
        s = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=g,
                              method="decision_function")
        auc = roc_auc_score(y, s)
        out[lab] = auc
        bins = np.linspace(np.percentile(s, 0.5), np.percentile(s, 99.5), 40)
        ax.hist(s[y == 0], bins=bins, color="#3f4652", alpha=.55, lw=0,
                label="Dense")
        ax.hist(s[y == 1], bins=bins, color=c, alpha=.62, lw=0, label=nice)
        ax.set_title(f"{nice}   AUC {auc:.2f}", pad=6)
        ax.set_yticks([])
        ax.set_xlabel("probe score")
        ax.legend(frameon=False, loc="upper left", handlelength=1.1,
                  borderpad=.2, labelspacing=.25)
    fig.tight_layout(pad=0.6)
    p = a.out or os.path.join(os.path.dirname(a.npz), "probe_vs_dense.pdf")
    fig.savefig(p, bbox_inches="tight")
    png = os.path.splitext(p)[0] + ".png"
    fig.savefig(png, bbox_inches="tight")
    print(f"wrote {p} and {png}")
    print(f"same-distribution floor (dense split in half): AUC {a.floor:.3f}")
    for lab, auc in out.items():
        print(f"  dense vs {lab:<14} AUC = {auc:.4f}  (+{auc - a.floor:.3f})")


if __name__ == "__main__":
    main()
