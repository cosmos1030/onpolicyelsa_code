import argparse
import json
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


DATA = Path("plotdata/plotdata")
# The self_gen file is swappable: the default is 6 prompts x 96 rollouts, and
# build_n30_npz.py writes a 30 x 64 companion. Panel (a) does not follow it --
# fixed-CoT displacement has its own 6 traces and is unaffected.
SELF_NPZ = "self_gen/pooled_L18_late_fp16.npz"
OUTPUT_PATH = Path("figures/divergence.pdf")
SPARSITIES = ["s50", "s60", "s70"]
XS = [50, 60, 70]
# noopd55, not noopd: the 0.33/0.33/0 w/o-OPD runs dropped OPD without
# renormalising the remaining two terms, so their total loss shrank to 2/3 and
# "no OPD" was confounded with a lower effective learning rate. The 0.5/0.5/0
# replacements are the clean ablation and carry the noopd55 label everywhere.
METHODS = [
    ("alps", "ALPS", "#4C78A8", "o"),
    ("sparsegpt", "SparseGPT", "#F58518", "s"),
    # Vega-10 green, the same palette the other three colours come from.
    # The muted purple it replaces disappeared against the red in the t-SNE.
    ("noopd55", "SCOUT w/o OPD", "#54A24B", "^"),
    ("ours", "SCOUT (Ours)", "#E45756", "D"),
]
# Dashed for the ablation, so the two SCOUT rows read as a pair in the line
# panels without relying on colour alone.
DASHED = {"noopd55"}
ALL_METHODS = METHODS


def tsne_models(methods):
    return [("dense", "Dense", "#7F7F7F", "o")] + [
        (f"{key}:s70", label, color, marker) for key, label, color, marker in methods
    ]


def style(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", linestyle=":", linewidth=0.7, color="#D0D0D0")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=1.1, length=3.5, labelsize=10.0)


def displacement(methods, layer="L18"):
    raw = json.loads((DATA / "fixed_cot/cot_displacement.json").read_text())
    out = {}
    for key, _, _, _ in methods:
        means, sems = [], []
        for sp in SPARSITIES:
            values = raw[f"{layer}/{key}_{sp}"]
            means.append(st.mean(values))
            sems.append(st.stdev(values) / len(values) ** 0.5)
        out[key] = (means, sems)
    return out


def mmd2(a, b, med):
    z = np.concatenate([a, b])
    sq = (z * z).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (z @ z.T), 0.0)
    kernel = np.exp(-d2 / med)
    n = len(a)
    return float(
        kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2.0 * kernel[:n, n:].mean()
    )


# The median-heuristic bandwidth is taken over the pooled embeddings, so adding
# a method to the figure would move it and every previously reported MMD with
# it. Pin the pool to the three methods the published numbers were computed
# from; the ablation is then measured against that same kernel rather than
# redefining the scale. All four curves still share one bandwidth per prompt,
# which is the property the panel needs.
BANDWIDTH_METHODS = ("alps", "sparsegpt", "ours")


def n_prompts(store):
    return len({k.split("|")[0] for k in store.files})


def divergence(methods):
    """MMD^2 to dense, with one bandwidth per prompt shared across methods.

    BANDWIDTH_METHODS does not follow `methods`: dropping ALPS and SparseGPT
    from a figure should change which curves are drawn, not what the y axis
    means. Pinning the kernel keeps every variant of this plot on one scale.
    """
    store = np.load(DATA / SELF_NPZ)
    P = n_prompts(store)
    get = lambda pi, key: store[f"p{pi}|{key}|L18|late"].astype(np.float32)
    out = {key: ([], []) for key, _, _, _ in methods}
    for key, _, _, _ in methods:
        for sp in SPARSITIES:
            names = ["dense"] + [f"{m}:{sp}" for m in BANDWIDTH_METHODS]
            values = []
            for pi in range(P):
                z = np.concatenate([get(pi, n) for n in names])
                sq = (z * z).sum(1)
                d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (z @ z.T), 0.0)
                med = np.median(d2[d2 > 0])
                values.append(mmd2(get(pi, "dense"), get(pi, f"{key}:{sp}"), med))
            out[key][0].append(st.mean(values))
            out[key][1].append(st.stdev(values) / len(values) ** 0.5)
    return out


def draw(axis, data, title, ylabel, methods):
    for key, label, color, marker in methods:
        means, sems = data[key]
        axis.errorbar(
            XS,
            means,
            yerr=sems,
            color=color,
            marker=marker,
            markersize=4.4,
            linewidth=2.0 if key == "ours" else 1.5,
            linestyle="--" if key in DASHED else "-",
            capsize=2.6,
            elinewidth=1.0,
            label=label,
        )
    axis.set_title(title, fontsize=11.0, fontweight="bold", pad=6)
    axis.set_xlabel("Unstructured sparsity (%)", fontsize=10.0)
    axis.set_ylabel(ylabel, fontsize=10.0)
    axis.set_xticks(XS, [f"{x}%" for x in XS])
    axis.set_xlim(47, 73)
    style(axis)


def tsne_panel(axis, pi, methods, legend=False):
    models = tsne_models(methods)
    store = np.load(DATA / SELF_NPZ)
    blocks = [
        store[f"p{pi}|{key}|L18|late"].astype(np.float32)
        for key, _, _, _ in models
    ]
    xy = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=0,
    ).fit_transform(np.concatenate(blocks))
    start = 0
    for block, (_, label, color, marker) in zip(blocks, models):
        points = xy[start:start + len(block)]
        start += len(block)
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=4.5,
            c=color,
            marker=marker,
            alpha=0.75,
            linewidths=0.0,
            label=label,
        )
    axis.set_xticks([])
    axis.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        axis.spines[side].set_linewidth(0.9)
        axis.spines[side].set_color("#BBBBBB")
    if legend:
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(0.0, -0.03),
            ncol=len(models),
            frameon=False,
            fontsize=7.8,
            handlelength=1.0,
            markerscale=2.2,
            columnspacing=1.2,
        )


def main():
    global SELF_NPZ
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="alps,sparsegpt,noopd55,ours",
                    help="comma-separated keys, in draw order")
    ap.add_argument("--out", default=str(OUTPUT_PATH))
    ap.add_argument("--self_npz", default=SELF_NPZ,
                    help="self_gen npz under plotdata/plotdata "
                         "(e.g. self_gen/pooled_L18_late_fp16_n30.npz)")
    args = ap.parse_args()
    SELF_NPZ = args.self_npz
    want = [k for k in args.methods.split(",") if k]
    known = {k: t for t in ALL_METHODS for k in (t[0],)}
    missing = [k for k in want if k not in known]
    if missing:
        raise SystemExit(f"unknown method(s): {missing}; have {list(known)}")
    methods = [known[k] for k in want]
    out_path = Path(args.out)
    # The published panels are tuned to the four-method ranges. Any other
    # subset gets autoscaled instead, because reusing those limits would
    # squash two near-flat curves into the bottom axis.
    full = want == [t[0] for t in ALL_METHODS]

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.3})
    fig = plt.figure(figsize=(11.9, 2.75))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.66], wspace=0.30)

    axis_a = fig.add_subplot(grid[0, 0])
    draw(axis_a, displacement(methods), "(a) Fixed reasoning trace",
         "Representation displacement", methods)
    if full:
        axis_a.set_ylim(0, 7.4)
    else:
        axis_a.set_ylim(0, axis_a.get_ylim()[1])
    axis_a.legend(loc="upper left", frameon=False, fontsize=8.6, handlelength=1.5)

    axis_b = fig.add_subplot(grid[0, 1])
    draw(axis_b, divergence(methods), "(b) Self-generated responses",
         r"MMD$^2$ to dense", methods)
    if full:
        axis_b.set_ylim(0, 1.12)
    else:
        axis_b.set_ylim(0, axis_b.get_ylim()[1])

    cells = grid[0, 2].subgridspec(1, 4, wspace=0.10)
    tsne_axes = []
    for index in range(4):
        axis = fig.add_subplot(cells[0, index])
        tsne_panel(axis, index, methods, legend=(index == 0))
        tsne_axes.append(axis)

    # Equalise the visual gaps: (a)->(b) contains (b)'s ylabel and ticks, while
    # (b)->(c) contains nothing, so equal gridspec spacing looks uneven. Shift
    # the (c) block left until the whitespace matches, then centre its title.
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted().transform_bbox
    tight = lambda ax: to_fig(ax.get_tightbbox(fig.canvas.get_renderer()))
    visual_gap = tight(axis_b).x0 - tight(axis_a).x1
    shift = tight(tsne_axes[0]).x0 - (tight(axis_b).x1 + visual_gap)
    grow = shift / len(tsne_axes)
    for index, axis in enumerate(tsne_axes):
        box = axis.get_position()
        axis.set_position(
            [box.x0 - shift + index * grow, box.y0, box.width + grow, box.height]
        )

    fig.canvas.draw()
    left = tsne_axes[0].get_position().x0
    right = tsne_axes[-1].get_position().x1
    fig.text(
        0.5 * (left + right),
        axis_b.get_position().y1 + 0.055,
        "(c) Generated responses at 70% sparsity",
        ha="center",
        va="bottom",
        fontsize=11.0,
        fontweight="bold",
    )

    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
