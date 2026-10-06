import csv
from pathlib import Path

import matplotlib.pyplot as plt


DATA_PATH = Path("scout_ablation_iclr2027/data/projection_events.csv")
OUTPUT_PATH = Path("figures/pace_matched_kl.pdf")
ARMS = {
    "scout_s70_d0.02": ("SCOUT", "#3b6ea8"),
    "sched_matched_s70": ("Cubic schedule", "#e07a2f"),
}


def load_events():
    events = {run: [] for run in ARMS}
    with DATA_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            run = row["run"]
            if run in events and row["phase"] == "growth":
                events[run].append(row)
    return events


def support_series(rows):
    steps = [0]
    sparsities = [0.0]
    for index, row in enumerate(rows):
        steps.append(int(row["step"]))
        if index + 1 < len(rows):
            sparsities.append(float(rows[index + 1]["pre_sparsity"]))
        else:
            sparsities.append(float(row["target_sparsity"]))
    return steps, sparsities


def main():
    events = load_events()
    fig, (growth_ax, kl_ax) = plt.subplots(1, 2, figsize=(10.5, 3.25))

    for run, (label, color) in ARMS.items():
        steps, sparsities = support_series(events[run])
        growth_ax.step(steps, sparsities, where="post", color=color, linewidth=1.8, label=label)

    for run in ("scout_s70_d0.02", "sched_matched_s70"):
        label, color = ARMS[run]
        values = [float(row["kl_at_k"]) / float(row["delta"]) for row in events[run]]
        kl_ax.plot(
            range(1, len(values) + 1),
            values,
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=3.7,
            label=label,
        )
    kl_ax.axhline(1.0, color="#222222", linestyle="--", linewidth=1.1, label="Trust radius")

    growth_ax.set_title("(a) Matched support-growth trajectory", fontsize=11.5, fontweight="bold", pad=6)
    growth_ax.set_xlabel("Optimizer step", fontsize=10.5)
    growth_ax.set_ylabel("Unstructured sparsity", fontsize=10.5)
    growth_ax.set_xlim(0, 250)
    growth_ax.set_ylim(0, 0.74)
    growth_ax.set_xticks([0, 50, 100, 150, 200, 250])
    growth_ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.7])

    kl_ax.set_title("(b) Projection displacement", fontsize=11.5, fontweight="bold", pad=6)
    kl_ax.set_xlabel("Growth projection event", fontsize=10.5)
    kl_ax.set_ylabel(r"Measured $D_t/\delta$", fontsize=10.5)
    kl_ax.set_xlim(0.5, 29.5)
    kl_ax.set_ylim(0.0, 3.0)
    kl_ax.set_xticks([1, 5, 10, 15, 20, 25, 29])
    kl_ax.set_yticks([0, 1, 2, 3])

    for axis in (growth_ax, kl_ax):
        axis.tick_params(labelsize=9.5)
        axis.grid(axis="y", color="#d0d0d0", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([], [], color=color, linewidth=1.8, label=label)
        for label, color in ARMS.values()
    ]
    handles.append(plt.Line2D([], [], color="#222222", linestyle="--", linewidth=1.1, label="Trust radius"))
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=4,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.25,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.21, top=0.75, wspace=0.32)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")


if __name__ == "__main__":
    main()
