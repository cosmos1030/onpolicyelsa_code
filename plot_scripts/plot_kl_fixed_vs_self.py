"""KL(dense || sparse) on fixed text and on the model's own text.

The scalar companion to the probe figure. The probe asks whether a sparse
model's rollouts are linearly separable from dense's; this asks how much
probability mass the sparse model puts in the wrong place, token by token, and
it can be read at two places:

  fixed  the dataset's reasoning trace for the prompt -- the text off-policy
         KD computes its targets on
  self   the model's own rollout for the SAME prompt

The claim is the GAP, not the level. "SCOUT has lower self-generated KL" invites
the objection that on-policy distillation optimises approximately that. "w/o OPD
matches dense on fixed text and comes apart on its own output" does not: the
fixed level is where the two arms are indistinguishable, so the difference
cannot be read as OPD simply optimising its own objective.

Curves are the mean over prompts of each prompt's KL binned into 16 equal
position buckets, so continuations of different length line up. Bands are the
standard error over PROMPTS (n=30).

Usage: plot_kl_fixed_vs_self.py --arms ours:s70=kl_951041.json,noopd55:s70=kl_951044.json
"""
import argparse
import json
import os

import numpy as np

DIR = ("/home1/doyoonkim/projects/elsa/logs/policy_divergence/"
       "kl_fixed_vs_selfgen")
NICE = {"ours": ("SCOUT", "#0f8b7e"), "noopd55": ("SCOUT w/o OPD", "#54A24B"),
        "kdonly": ("SCOUT w/o OPD", "#54A24B"), "opdonly": ("OPD only", "#b08600"),
        "alps_sft": ("ALPS+retrain", "#d95f0e")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--arms", default="ours:s70=kl_951041.json,"
                                      "noopd55:s70=kl_951044.json")
    ap.add_argument("--style", default="overlay",
                    choices=["overlay", "gap", "curves"],
                    help="overlay: one panel, both arms' fixed curves (they "
                         "coincide) plus each arm's own-rollout curve -- shows "
                         "the levels AND that the fixed levels match. gap: the "
                         "paired difference alone. curves: two panels.")
    ap.add_argument("--xunit", default="tokens", choices=["tokens", "percent"],
                    help="tokens: bucket centres in absolute token position. "
                         "Valid because both sides are cut to --max_new, so a "
                         "bucket is the same 128-token span on either side; a "
                         "continuation shorter than the cap is stretched over "
                         "the same 16 buckets (mean length 2032/2048 here).")
    ap.add_argument("--max_new", type=int, default=2048,
                    help="the generation budget the runs used; sets the token "
                         "axis")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
        "legend.fontsize": 8.5, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 150,
    })
    arms = [kv.split("=", 1) for kv in a.arms.split(",") if kv]
    if a.xunit == "tokens":
        # bucket centres: 16 equal bins over the budget -> 128 tokens each
        XS = (np.arange(16) + 0.5) * a.max_new / 16.0
        XLAB = "token position in continuation"
    else:
        XS = np.linspace(0, 100, 16)
        XLAB = "position in continuation (%)"

    if a.style == "overlay":
        # Four curves, three visible. KL(dense || sparse) is measured against
        # a particular sparse model, so "on fixed text" is not one shared
        # quantity -- there is one such curve per arm, and they are drawn
        # separately here. That they land on top of each other (0.1849 vs
        # 0.1848 on average) is the result, not a shortcut: if the arms
        # differed on the dataset trace, the separation between the solid
        # curves could be read as "w/o OPD is just a worse model" rather than
        # as exposure bias.
        fig, ax = plt.subplots(figsize=(4.2, 2.7))
        x = XS

        def band(C, color, ls, lw, lab, z=3, al=.15):
            m = np.nanmean(C, 0)
            se = np.nanstd(C, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(C), 0))
            ax.plot(x, m, ls, color=color, lw=lw, label=lab, zorder=z)
            ax.fill_between(x, m - se, m + se, color=color, alpha=al, lw=0,
                            zorder=z - 1)

        recs = []
        for key, fn in arms:
            recs.append((key, json.load(open(os.path.join(a.dir, fn)))[key]))
        for i, (key, rec) in enumerate(recs):
            band(np.array(rec["fixed_curve"], float), "#8b8f96", "--", 1.6,
                 "fixed text: both arms (curves coincide)" if i == 0 else None, z=2, al=.12)
        for key, rec in recs:
            nice, c = NICE.get(key.split(":")[0], (key, "#888"))
            band(np.array(rec["self_curve"], float), c, "-", 1.9,
                 f"{nice} \u2014 own rollouts", z=4)
        ax.set_xlabel(XLAB)
        ax.set_ylabel("KL(dense $\\|$ sparse)")
        ax.legend(frameon=False, loc="upper right", handlelength=1.6,
                  borderpad=.2, labelspacing=.3)
        fig.tight_layout(pad=0.6)
        p = a.out or os.path.join(a.dir, "kl_overlay.pdf")
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(os.path.splitext(p)[0] + ".png", bbox_inches="tight")
        print(f"wrote {p} (+ .png)")
        for key, rec in recs:
            s_ = rec["summary"]
            print(f"  {key:<14} fixed {s_['fixed_mean']:.4f}  self {s_['self_mean']:.4f}"
                  f"  gap {s_['gap_mean']:+.4f} (t={s_['gap_t']:+.2f})")
        return

    if a.style == "gap":
        # One panel, the paired difference. Reading two overlaid levels and
        # subtracting them by eye is what made the two-panel version
        # unreadable; the difference is the claim, so plot the difference.
        fig, ax = plt.subplots(figsize=(3.9, 2.6))
        x = XS
        for key, fn in arms:
            rec = json.load(open(os.path.join(a.dir, fn)))[key]
            nice, c = NICE.get(key.split(":")[0], (key, "#888"))
            # paired within a prompt, then averaged -- prompts differ in
            # difficulty and that cancels in the difference
            Dm = np.array(rec["self_curve"], float) - np.array(rec["fixed_curve"], float)
            m = np.nanmean(Dm, 0)
            se = np.nanstd(Dm, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(Dm), 0))
            s_ = rec["summary"]
            ax.plot(x, m, "-", color=c, lw=1.9,
                    label=f"{nice}  ({s_['gap_mean']:+.3f}, t={s_['gap_t']:.1f})")
            ax.fill_between(x, m - se, m + se, color=c, alpha=.18, lw=0)
        ax.axhline(0, color="#444", lw=1.0, ls=":")
        ax.set_xlabel(XLAB)
        ax.set_ylabel("KL on own rollouts $-$ KL on fixed text")
        ax.legend(frameon=False, loc="upper right", handlelength=1.4,
                  borderpad=.2, labelspacing=.3)
        fig.tight_layout(pad=0.6)
        p = a.out or os.path.join(a.dir, "kl_gap.pdf")
        fig.savefig(p, bbox_inches="tight")
        fig.savefig(os.path.splitext(p)[0] + ".png", bbox_inches="tight")
        print(f"wrote {p} (+ .png)")
        for key, fn in arms:
            s_ = json.load(open(os.path.join(a.dir, fn)))[key]["summary"]
            print(f"  {key:<14} fixed {s_['fixed_mean']:.4f}  self {s_['self_mean']:.4f}"
                  f"  gap {s_['gap_mean']:+.4f} (t={s_['gap_t']:+.2f})")
        return

    fig, axes = plt.subplots(1, len(arms), figsize=(3.4 * len(arms), 2.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (key, fn) in zip(axes, arms):
        rec = json.load(open(os.path.join(a.dir, fn)))[key]
        nice, c = NICE.get(key.split(":")[0], (key, "#888"))
        x = XS
        for name, col, ls, lab in [("fixed_curve", "#8b8f96", "--", "fixed text"),
                                   ("self_curve", c, "-", "own rollouts")]:
            C = np.array(rec[name], float)
            m = np.nanmean(C, 0)
            se = np.nanstd(C, 0, ddof=1) / np.sqrt(np.sum(~np.isnan(C), 0))
            ax.plot(x, m, ls, color=col, lw=1.8, label=lab)
            ax.fill_between(x, m - se, m + se, color=col, alpha=.18, lw=0)
        s = rec["summary"]
        ax.set_title(f"{nice}   gap {s['gap_mean']:+.3f} "
                     f"(t={s['gap_t']:.1f})", pad=6)
        ax.set_xlabel(XLAB)
        ax.legend(frameon=False, loc="upper left", handlelength=1.6,
                  borderpad=.2, labelspacing=.25)
    axes[0].set_ylabel("KL(dense $\\|$ sparse)")
    fig.tight_layout(pad=0.6)
    p = a.out or os.path.join(a.dir, "kl_fixed_vs_self.pdf")
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(os.path.splitext(p)[0] + ".png", bbox_inches="tight")
    print(f"wrote {p} (+ .png)")
    for key, fn in arms:
        s = json.load(open(os.path.join(a.dir, fn)))[key]["summary"]
        print(f"  {key:<14} fixed {s['fixed_mean']:.4f}  self {s['self_mean']:.4f}"
              f"  gap {s['gap_mean']:+.4f} (t={s['gap_t']:+.2f})")


if __name__ == "__main__":
    main()
