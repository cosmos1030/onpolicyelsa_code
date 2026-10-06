"""Read the coverage embeddings and report what they can actually support.

Two numbers, both relative to dense, both with prompt-level uncertainty:

  paired cosine distance   for each prompt i and window k, the mean cosine
                           distance from the reference-CoT window to that
                           model's rollout windows at the SAME depth. Averaged
                           within a prompt, then reported as the increase over
                           dense.
  linear-probe AUC         how separable fixed and rollout are, again as the
                           increase over dense.

The absolute value of either is not evidence of anything. Reference text and
generated text differ for reasons that have nothing to do with pruning -- an
earlier version of this experiment drew the fixed side from random corpus
documents and every model, dense included, separated at AUC 0.994-0.999. What
carries a claim is the gap to dense:

    delta_model = D(F, R_model) - D(F, R_dense)

If that grows with sparsity, pruning is moving the model away from the text it
is trained on. If it is flat, the separation is a property of "reference vs
generated" and says nothing about pruning.

Uncertainty is over PROMPTS (n=30), never over windows. Windows from one
trajectory are strongly correlated, so treating 1,640 of them as independent
would shrink the interval by roughly sqrt(8) for free.

No permutation test. The natural null would shuffle the source label, but any
shuffle fine-grained enough to be cheap also breaks the within-trajectory
correlation it is supposed to respect; the bootstrap CI answers the same
question without that problem.

Usage: coverage_postprocess.py --npz coverage_<jobid>.npz [--dense dense]
"""
import argparse

import numpy as np


def cos_dist(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return 1.0 - A @ B.T


def paired_by_prompt(fixed, gF, roll, gR):
    """Mean cosine distance from each prompt's fixed windows to its own
    rollout windows. Returns {prompt: distance}."""
    out = {}
    for p in np.unique(gF):
        F = fixed[gF == p]
        R = roll[gR == p]
        if len(F) == 0 or len(R) == 0:
            continue
        # Both sides were cut to the same span and read at identical offsets,
        # so row k of F and row k of each rollout are the same depth. Comparing
        # all pairs would mix depths back in, so align by window index.
        n = min(len(F), len(R))
        d = []
        for k in range(len(F)):
            same = R[k::len(F)] if len(R) % len(F) == 0 else R
            d.append(cos_dist(F[k:k + 1], same).mean())
        out[int(p)] = float(np.mean(d))
    return out


def boot(vals, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    s = [rng.choice(v, len(v), replace=True).mean() for _ in range(n)]
    return float(v.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def probe_auc(fixed, gF, roll, gR, seed=0):
    """Grouped-CV AUC with the class imbalance handled.

    fixed contributes ~7 windows per prompt and each model ~8 rollouts x 7,
    i.e. 8:1. AUC tolerates imbalanced *test* sets, but an 8:1 training
    objective still tilts the boundary, so the rollout side is subsampled to
    the fixed side's size, per prompt.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    Rs, gs = [], []
    for p in np.unique(gF):
        idx = np.where(gR == p)[0]
        k = min(len(idx), int((gF == p).sum()))
        if k == 0:
            continue
        pick = rng.choice(idx, k, replace=False)
        Rs.append(roll[pick]); gs.append(np.full(k, p))
    if not Rs:
        return float("nan")
    R = np.concatenate(Rs); gR2 = np.concatenate(gs)
    X = np.concatenate([fixed, R]).astype(np.float32)
    y = np.r_[np.zeros(len(fixed)), np.ones(len(R))]
    g = np.r_[gF, gR2]
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced",
                                           random_state=0))
    return float(cross_val_score(clf, X, y, cv=GroupKFold(5), groups=g,
                                 scoring="roc_auc").mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--dense", default="dense")
    a = ap.parse_args()

    z = np.load(a.npz)
    labels = [k for k in z.files if not k.startswith("owner_") and k != "fixed"]
    F = z["fixed"].astype(np.float32)
    gF = z["owner_fixed"]
    print(f"fixed: {F.shape}, {len(np.unique(gF))} prompts\n")

    per = {}
    for lab in labels:
        per[lab] = paired_by_prompt(F, gF, z[lab].astype(np.float32),
                                    z[f"owner_{lab}"])

    if a.dense not in per:
        print(f"!! no '{a.dense}' arm; absolute values only, which prove nothing")
        base = None
    else:
        base = per[a.dense]

    print(f"{'model':<18}{'paired cos d':>14}{'95% CI':>22}"
          f"{'vs dense':>11}{'95% CI':>22}{'AUC':>8}{'vs dense':>10}")
    print("-" * 105)
    auc = {}
    for lab in labels:
        auc[lab] = probe_auc(F, gF, z[lab].astype(np.float32), z[f"owner_{lab}"])
    for lab in labels:
        ps = sorted(per[lab])
        m, lo, hi = boot([per[lab][p] for p in ps])
        if base is None or lab == a.dense:
            dm = dlo = dhi = float("nan")
        else:
            d = [per[lab][p] - base[p] for p in ps if p in base]
            dm, dlo, dhi = boot(d)
        da = float("nan") if base is None else auc[lab] - auc[a.dense]
        print(f"{lab:<18}{m:14.4f}  [{lo:.4f}, {hi:.4f}]"
              f"{dm:+11.4f}  [{dlo:+.4f}, {dhi:+.4f}]{auc[lab]:8.3f}{da:+10.3f}")

    print("\nCI is over prompts (n=%d), not windows." % len(np.unique(gF)))
    print("Read the 'vs dense' columns. An interval that excludes 0 means this")
    print("model's rollouts sit further from the reference text than dense's do;")
    print("the absolute columns cannot separate that from 'generated text just")
    print("differs from written text'.")


if __name__ == "__main__":
    main()
