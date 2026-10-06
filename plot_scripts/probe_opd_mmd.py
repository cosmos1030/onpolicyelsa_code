"""Is OPD's vanishing MMD gap at s70 a fact about the models or about the kernel?

Two confounds are cheap to rule out with data already on disk.

1. Bandwidth drift. The median heuristic is taken over dense + ALPS + SparseGPT
   + SCOUT pooled together. At s70 ALPS and SparseGPT sit far away, which pushes
   the median distance up, which widens the Gaussian, which compresses MMD
   between two clouds that are already close -- exactly the SCOUT / SCOUT-w/o-OPD
   pair. So the scale the panel is drawn on is itself a function of sparsity.
   Three bandwidth rules are computed here:

     pooled   the current one, per sparsity
     dense    median within the dense cloud only -- no dependence on which
              methods are in the figure, and none on sparsity
     s50fix   the pooled value computed at s50 and reused at s60 and s70, so
              one number covers the whole x axis

   If the s70 difference reappears under `dense` or `s50fix`, the disappearance
   was a scale artefact, not a property of the models.

2. Where in the rollout, and how deep. The late window averages continuation
   tokens 1024-2048 -- the middle of the reasoning, long after the branch points
   that decide the answer. The pool also holds the early window (0-512) and
   layer 36. If OPD separates in early/L36 but not late/L18, the effect lives
   somewhere the current panel does not look.

The test is paired: both models are measured against dense on the SAME prompt
with the SAME bandwidth, and the per-prompt differences are averaged. Pairing is
what makes n=30 informative -- prompt-to-prompt variation in MMD dwarfs the
effect being measured.

Usage: probe_opd_mmd.py [--pool DIR]
"""
import argparse
import glob
import os

import numpy as np

POOL = "/home1/doyoonkim/projects/elsa/logs/policy_divergence/n30_k64_core"
SPARSITIES = ["s50", "s60", "s70"]
BW_POOL = ("alps", "sparsegpt", "ours")   # what the figure's bandwidth uses


def sqdist(z):
    sq = (z * z).sum(1)
    return np.maximum(sq[:, None] + sq[None, :] - 2.0 * (z @ z.T), 0.0)


def mmd2(a, b, med):
    z = np.concatenate([a, b])
    k = np.exp(-sqdist(z) / med)
    n = len(a)
    return float(k[:n, :n].mean() + k[n:, n:].mean() - 2.0 * k[:n, n:].mean())


def paired(diffs):
    d = np.asarray(diffs)
    sem = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), sem, (d.mean() / sem if sem > 0 else np.nan), int((d > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL)
    a = ap.parse_args()

    z = np.load(os.path.join(a.pool, "pooled.npz"))
    side = [np.load(f) for f in sorted(glob.glob(os.path.join(a.pool,
                                                              "pooled_extra_*.npz")))]

    def fetch(k):
        if k in z.files:
            return z[k]
        for e in side:
            if k in e.files:
                return e[k]
        return None

    P = len({k.split("|")[0] for k in z.files})
    combos = [(L, w) for L in (18, 36) for w in ("late", "early")]

    for L, w in combos:
        g = lambda pi, lab: (None if (v := fetch(f"p{pi}|{lab}|L{L}|{w}")) is None
                             else v.astype(np.float32))
        if g(0, "dense") is None:
            print(f"\n### L{L}/{w}: not in the pool, skipped")
            continue

        # bandwidths, per prompt
        bw = {"pooled": {}, "dense": {}, "s50fix": {}}
        for pi in range(P):
            D = g(pi, "dense")
            d2 = sqdist(D)
            bw["dense"][pi] = float(np.median(d2[d2 > 0]))
            for sp in SPARSITIES:
                arrs = [D] + [v for m in BW_POOL
                              if (v := g(pi, f"{m}:{sp}")) is not None]
                dd = sqdist(np.concatenate(arrs))
                bw["pooled"][(pi, sp)] = float(np.median(dd[dd > 0]))
            for sp in SPARSITIES:
                bw["s50fix"][(pi, sp)] = bw["pooled"][(pi, "s50")]

        print(f"\n{'='*78}\n### layer {L}, {w} window   (n={P} prompts)\n{'='*78}")
        for rule in ("pooled", "dense", "s50fix"):
            print(f"\n-- bandwidth: {rule} --")
            print(f"{'':6}{'SCOUT':>10}{'w/o OPD':>10}{'diff':>10}"
                  f"{'SEM':>9}{'t':>8}{'wins':>8}")
            for sp in SPARSITIES:
                o, n_, d = [], [], []
                for pi in range(P):
                    med = (bw[rule][pi] if rule == "dense"
                           else bw[rule][(pi, sp)])
                    D = g(pi, "dense")
                    a_ = mmd2(D, g(pi, f"ours:{sp}"), med)
                    b_ = mmd2(D, g(pi, f"noopd55:{sp}"), med)
                    o.append(a_); n_.append(b_); d.append(b_ - a_)
                m, sem, t, wins = paired(d)
                print(f"{sp:6}{np.mean(o):10.4f}{np.mean(n_):10.4f}"
                      f"{m:+10.4f}{sem:9.4f}{t:+8.2f}{wins:>5}/{P}")


if __name__ == "__main__":
    main()
