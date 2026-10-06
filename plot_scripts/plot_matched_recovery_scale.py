from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = Path("figures/matched_recovery_scale.pdf")
SCALES = ["1.7B", "4B", "8B"]
SERIES = {
    "ALPS": ([4.52, 25.62, 32.96], "#2B6C9E"),
    "ALPS+recovery": ([21.76, 40.86, 46.42], "#4DAF4A"),
    "SCOUT": ([29.92, 45.58, 49.74], "#E45756"),
}
GRID = "#D0D0D0"


def main():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.4,
        }
    )
    positions = np.arange(len(SCALES))
    width = 0.24
    label_offsets = [-0.01, -0.025, 0.025]
    fig, axis = plt.subplots(figsize=(4.35, 2.8), facecolor="white")

    for index, (label, (values, color)) in enumerate(SERIES.items()):
        bars = axis.bar(
            positions + (index - 1) * width,
            values,
            width=width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.9,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2 + label_offsets[index],
                value + 1.2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold" if label == "SCOUT" else "normal",
            )

    axis.set_ylabel("Five-task Avg")
    axis.set_xlabel("Qwen3 model scale")
    axis.set_xticks(positions, SCALES)
    axis.set_ylim(0, 56)
    axis.set_yticks([0, 10, 20, 30, 40, 50])
    axis.grid(axis="y", linestyle=":", linewidth=0.8, color=GRID)
    axis.set_axisbelow(True)
    axis.tick_params(width=1.2, length=4)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.2, top=0.81)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
