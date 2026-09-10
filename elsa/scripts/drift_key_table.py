#!/usr/bin/env python
"""The one table that decides the drift claim.

Three ways of pooling the same states, each with the control that tells text
apart from representation:

  raw        every state its own row
  centered   each prompt's mean (pooled over both regimes) removed first, so
             problem identity cannot stand in for the regime label
  seqmean    one row per (prompt, regime): no within-sequence correlation left

and for each, two encoders on the SAME tokens:

  own        the pruned model reading its own rollouts
  dense      the dense model reading those same rollouts

If separability rises with sparsity under `own` but the `dense` column rises
just as fast, the rise is in the text, not in the representation, and the drift
reading is dead. The claim needs own - dense to grow.
"""
import argparse
import json
import os

import numpy as np


def probe_auc(Xa, ga, Xb, gb, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(Xa) < 10 or len(Xb) < 10:
        return float("nan")
    X = np.concatenate([Xa, Xb]).astype(np.float32)
    y = np.concatenate([np.zeros(len(Xa)), np.ones(len(Xb))])
    g = np.concatenate([ga, gb])
    n = min(5, len(set(ga.tolist())), len(set(gb.tolist())))
    if n < 2:
        return float("nan")
    s = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=n).split(X, y, g):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=seed))
        clf.fit(X[tr], y[tr])
        s[te] = clf.decision_function(X[te])
    return float(roc_auc_score(y, s))


def pooled_center(Xa, ga, Xb, gb):
    A, B = Xa.copy(), Xb.copy()
    for p in np.unique(np.concatenate([ga, gb])):
        ma, mb = ga == p, gb == p
        n = ma.sum() + mb.sum()
        if n:
            mu = (A[ma].sum(0) + B[mb].sum(0)) / n
            A[ma] -= mu
            B[mb] -= mu
    return A, B


def seq_means(X, g):
    ps = np.unique(g)
    return np.stack([X[g == p].mean(0) for p in ps]), ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.states_dir, "states_meta.json")))
    npz = np.load(os.path.join(args.states_dir, "states.npz"))
    keys = set(meta["keys"])
    labels = meta["labels"]

    def get(enc, cond, L, w):
        k = f"{enc}|{cond}|L{L}|{w}"
        if k not in keys:
            return None
        return np.asarray(npz[k], np.float32), np.asarray(npz[k + "|g"], int)

    out = {}
    for L in meta["layers"]:
        for w in meta["windows"]:
            print(f"\n{'='*78}\nlayer {L}, window {w}\n{'='*78}")
            print(f"{'':<6} " + "".join(f"{p:>26}" for p in ("raw", "centered", "seqmean")))
            print(f"{'':<6} " + "".join(f"{'own':>8}{'dense':>9}{'diff':>9}" for _ in range(3)))
            for lab in labels:
                F, S = get(lab, "fixed", L, w), get(lab, f"self:{lab}", L, w)
                if F is None or S is None:
                    continue
                DF = get("dense", "fixed", L, w)
                DS = get("dense", f"self:{lab}", L, w)
                row = f"{lab:<6} "
                for mode in ("raw", "centered", "seqmean"):
                    def prep(A, B):
                        if mode == "raw":
                            return A[0], A[1], B[0], B[1]
                        if mode == "centered":
                            Ca, Cb = pooled_center(A[0], A[1], B[0], B[1])
                            return Ca, A[1], Cb, B[1]
                        Ma, pa = seq_means(*A)
                        Mb, pb = seq_means(*B)
                        return Ma, pa, Mb, pb
                    a = probe_auc(*prep(F, S), seed=args.seed)
                    if lab == "dense" or DF is None or DS is None:
                        row += f"{a:8.3f}{'—':>9}{'—':>9}"
                        out[f"L{L}/{w}/{lab}/{mode}"] = {"own": a}
                        continue
                    d = probe_auc(*prep(DF, DS), seed=args.seed)
                    row += f"{a:8.3f}{d:9.3f}{a - d:+9.3f}"
                    out[f"L{L}/{w}/{lab}/{mode}"] = {"own": a, "dense": d, "diff": a - d}
                print(row, flush=True)

    p = os.path.join(args.states_dir, "key_table.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
