import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EVENTS_PATH = Path("scout_ablation_iclr2027/data/projection_events.csv")
OUTPUT_PATH = Path("figures/analysis.pdf")
COLORS = {
    "scout": "#E45756",
    "without_tr": "#8C564B",
    "without_refresh": "#9E9E9E",
    "cubic": "#E07A2F",
}

RUNS = [
    ("scout_s70_d0.02", "SCOUT", "scout"),
    ("sched_matched_s70", "Cubic schedule", "cubic"),
]


def style(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", linestyle=":", linewidth=0.7, color="#D0D0D0")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=1.1, length=3.5, labelsize=10.5)


def load_events():
    runs = {name: [] for name, _, _ in RUNS}
    with EVENTS_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["run"] in runs:
                runs[row["run"]].append(row)
    return runs


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.3})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.05))
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.19, top=0.86, wspace=0.27)

    axis = axes[0]
    scales = ["Qwen3-4B", "Qwen3-8B"]
    positions = np.arange(len(scales))
    width = 0.25
    series = [
        ("SCOUT (Ours)", [45.58, 50.13], COLORS["scout"]),
        ("w/o trust region", [36.55, 42.86], COLORS["without_tr"]),
        ("w/o rollout refresh", [43.97, 48.93], COLORS["without_refresh"]),
    ]
    for index, (label, values, color) in enumerate(series):
        bars = axis.bar(
            positions + (index - 1) * width,
            values,
            width,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            label=label,
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.7,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight="bold" if label.startswith("SCOUT") else "normal",
            )
    axis.set_title("(a) Component ablation at 70%", fontsize=11.5, fontweight="bold", pad=5)
    axis.set_ylabel("Five-task average", fontsize=10.5)
    axis.set_xticks(positions, scales)
    axis.set_xlim(-0.55, 1.55)
    axis.set_ylim(0, 72)
    axis.set_yticks([0, 10, 20, 30, 40, 50])
    axis.legend(loc="upper left", frameon=False, fontsize=8.4, handlelength=1.25)
    style(axis)

    axis = axes[1]
    events = load_events()
    for run, label, key in RUNS:
        points = [
            (int(row["step"]), float(row["pre_sparsity"]))
            for row in events[run]
            if int(row["step"]) <= 288
        ]
        axis.plot(
            [step for step, _ in points],
            [100.0 * value for _, value in points],
            color=COLORS[key],
            linewidth=1.7,
            marker="o",
            markersize=2.6,
            label=label,
        )
    axis.axhline(70.0, color="#333333", linestyle="--", linewidth=1.0, label="Target 70%")
    axis.set_title("(b) Sparsity trajectory", fontsize=11.5, fontweight="bold", pad=5)
    axis.set_xlabel("Training step", fontsize=10.5)
    axis.set_ylabel("Sparsity (%)", fontsize=10.5)
    axis.set_xlim(0, 288)
    axis.set_ylim(0, 82)
    axis.set_xticks([0, 64, 128, 192, 256])
    axis.set_yticks([0, 20, 40, 60, 70])
    axis.legend(loc="lower right", frameon=False, fontsize=9.0, handlelength=1.35)
    style(axis)

    axis = axes[2]
    for run, label, key in RUNS:
        values = [
            float(row["kl_at_k"]) / float(row["delta"])
            for row in events[run]
            if row["phase"] == "growth"
        ]
        axis.plot(
            range(1, len(values) + 1),
            values,
            color=COLORS[key],
            linewidth=1.6,
            marker="o",
            markersize=3.2,
            label=label,
        )
    axis.axhline(1.0, color="#333333", linestyle="--", linewidth=1.0, label="Trust radius")
    axis.set_title("(c) Projection displacement", fontsize=11.5, fontweight="bold", pad=5)
    axis.set_xlabel("Pre-target projection event", fontsize=10.5)
    axis.set_ylabel(r"Accepted KL $/\,\delta$", fontsize=10.5)
    axis.set_xlim(0.5, 29.5)
    axis.set_ylim(0, 3.0)
    axis.set_xticks([1, 5, 10, 15, 20, 25, 29])
    axis.set_yticks([0, 1, 2, 3])
    axis.legend(loc="upper left", frameon=False, fontsize=9.0, handlelength=1.35)
    style(axis)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
