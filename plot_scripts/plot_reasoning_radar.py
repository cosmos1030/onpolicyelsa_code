from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASKS = ["MATH", "LCB", "GPQA", "IFEval", "GSM8K"]
DISPLAY_METHODS = ["SparseGPT", "ALPS", "ALPS+retrain", "Ours"]
ANGLES = np.linspace(0, 2 * np.pi, len(TASKS), endpoint=False).tolist()
ANGLES += ANGLES[:1]
MIN_RING = 0.2
RINGS = [0.2, 0.4, 0.6, 0.8, 1.0]
COLUMN_HEADER_Y_OFFSET = 0.05
ROW_HEADER_X_OFFSET = 0.075
LABEL_RADIUS = {
    "MATH": 1.16,
    "LCB": 1.16,
    "GPQA": 1.16,
    "IFEval": 1.16,
    "GSM8K": 1.20,
}

FULL_DATA = {
    "Qwen3-1.7B": {
        "50%": {
            "SparseGPT": [54.8, 0.0, 28.8, 25.5, 70.7],
            "ALPS": [76.0, 8.2, 30.8, 50.8, 77.8],
            "ALPS+retrain": [72.8, 11.2, 29.8, 55.3, 72.7],
            "ELSA": [48.0, 3.7, 27.8, 25.5, 60.1],
            "Ours": [72.8, 7.46, 34.85, 52.13, 75.89],
        },
        "60%": {
            "SparseGPT": [0.2, 0.0, 0.0, 12.0, 30.2],
            "ALPS": [53.2, 0.0, 27.3, 23.7, 63.6],
            "ALPS+retrain": [61.8, 6.0, 29.3, 40.7, 69.5],
            "ELSA": [6.6, 0.4, 27.3, 18.9, 19.2],
            "Ours": [62.4, 2.99, 29.80, 38.08, 65.20],
        },
        "70%": {
            "SparseGPT": [0.0, 0.0, 2.5, 14.0, 0.0],
            "ALPS": [2.6, 0.0, 8.6, 11.3, 0.1],
            "ALPS+retrain": [26.6, 0.0, 26.8, 19.0, 36.4],
            "ELSA": [3.4, 0.0, 26.8, 13.7, 3.4],
            "Ours": [43.6, 1.49, 24.75, 25.88, 53.75],
        },
    },
    "Qwen3-4B": {
        "50%": {
            "SparseGPT": [80.4, 13.1, 38.9, 62.3, 80.4],
            "ALPS": [86.2, 20.1, 42.9, 67.8, 82.3],
            "ALPS+retrain": [87.4, 22.8, 42.4, 73.6, 80.0],
            "ELSA": [76.2, 13.4, 40.9, 57.9, 76.3],
            "Ours": [86.8, 26.1, 40.9, 72.6, 83.9],
        },
        "60%": {
            "SparseGPT": [52.0, 0.0, 30.3, 22.6, 72.0],
            "ALPS": [79.4, 10.4, 35.4, 41.8, 75.9],
            "ALPS+retrain": [81.2, 19.0, 32.8, 54.7, 77.0],
            "ELSA": [44.4, 3.0, 28.3, 29.6, 58.2],
            "Ours": [81.2, 18.3, 41.4, 56.2, 79.7],
        },
        "70%": {
            "SparseGPT": [14.4, 0.0, 26.3, 11.6, 21.6],
            "ALPS": [35.8, 0.0, 25.8, 16.3, 50.2],
            "ALPS+retrain": [66.2, 9.0, 26.8, 31.4, 70.9],
            "ELSA": [28.8, 0.4, 27.3, 20.5, 41.5],
            "Ours": [72.4, 8.2, 35.9, 39.9, 71.5],
        },
    },
}

COLORS = {
    "SparseGPT": "#4c78a8",
    "ALPS": "#f58518",
    "ALPS+retrain": "#b279a2",
    "Ours": "#54a24b",
}


def close(values):
    return values + values[:1]


def axis_bounds(panel):
    lowers = []
    uppers = []
    for task_idx in range(len(TASKS)):
        values = [scores[task_idx] for scores in panel.values()]
        lower = min(values)
        upper = max(values)
        if upper - lower < 1e-6:
            upper = lower + 1.0
        lowers.append(lower)
        uppers.append(upper)
    return lowers, uppers


def normalize(values, lowers, uppers):
    out = []
    for idx, value in enumerate(values):
        frac = (value - lowers[idx]) / (uppers[idx] - lowers[idx])
        out.append(MIN_RING + frac * (1.0 - MIN_RING))
    return out


def ring_value(lower, upper, ring):
    frac = (ring - MIN_RING) / (1.0 - MIN_RING)
    frac = min(max(frac, 0.0), 1.0)
    return lower + frac * (upper - lower)


def draw_task_labels(ax):
    for angle, task in zip(ANGLES[:-1], TASKS):
        radius = LABEL_RADIUS[task]
        if task == "MATH":
            ha = "center"
        elif task in {"LCB", "GPQA"}:
            ha = "left"
        else:
            ha = "right"
        ax.text(
            angle,
            radius,
            task,
            fontsize=11,
            fontweight="bold",
            color="#222222",
            ha=ha,
            va="center",
        )


def draw_ring_labels(ax, lowers, uppers):
    for angle, lower, upper in zip(ANGLES[:-1], lowers, uppers):
        for ring in RINGS:
            value = ring_value(lower, upper, ring)
            ax.text(
                angle,
                ring,
                f"{value:.0f}",
                fontsize=7.5,
                fontweight="bold",
                color="#666666",
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.08},
            )


def main():
    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(
        2,
        3,
        subplot_kw={"projection": "polar"},
        figsize=(12.2, 7.6),
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.16, wspace=0.28, hspace=0.34)

    sparsities = ["50%", "60%", "70%"]
    models = list(FULL_DATA.keys())

    for row, model in enumerate(models):
        for col, sparsity in enumerate(sparsities):
            ax = axes[row, col]
            panel = FULL_DATA[model][sparsity]
            lowers, uppers = axis_bounds(panel)

            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(ANGLES[:-1])
            ax.set_xticklabels([])
            ax.set_ylim(0, 1)
            ax.set_yticks(RINGS)
            ax.set_yticklabels([])
            ax.grid(color="#bbbbbb", alpha=0.6, linewidth=0.6)
            ax.spines["polar"].set_color("#999999")
            ax.spines["polar"].set_linewidth(0.8)

            draw_task_labels(ax)
            draw_ring_labels(ax, lowers, uppers)

            for method in DISPLAY_METHODS:
                values = close(normalize(panel[method], lowers, uppers))
                ax.plot(ANGLES, values, color=COLORS[method], linewidth=1.8, label=method)
                ax.fill(ANGLES, values, color=COLORS[method], alpha=0.08)

    # Column headers
    for col, sparsity in enumerate(sparsities):
        pos = axes[0, col].get_position()
        fig.text(
            pos.x0 + pos.width / 2,
            pos.y1 + COLUMN_HEADER_Y_OFFSET,
            sparsity,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="#222222",
        )

    # Row headers
    for row, model in enumerate(models):
        pos = axes[row, 0].get_position()
        fig.text(
            pos.x0 - ROW_HEADER_X_OFFSET,
            pos.y0 + pos.height / 2,
            model,
            ha="right",
            va="center",
            fontsize=15,
            fontweight="bold",
            color="#222222",
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.045),
        fontsize=11,
    )

    fig.savefig(out_dir / "reasoning_radar.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "reasoning_radar.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
