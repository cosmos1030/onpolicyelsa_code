#!/usr/bin/env python
"""Ways of showing fixed-vs-self drift that are not t-SNE.

t-SNE was the wrong instrument here. It preserves local neighbourhoods, and the
neighbourhoods in this data are organised by which problem a state came from,
not by which regime produced it -- so a direction a linear probe finds at AUC
0.96 leaves no visible mark on the map. Everything below either measures the
departure directly or projects along a direction chosen to show it.

  depth_auc     cross-fitted probe AUC as a function of token depth, one line
                per model, with the dense-encoder control dashed underneath.
                Drift is a claim about depth, so this is the shape that matters:
                does the gap open up as the model generates further?
  depth_mmd     the same sweep with kernel MMD instead of a probe, against a
                paired permutation null -- no classifier, no fitting.
  effrank       participation ratio of each state cloud. If the pruned model
                falls into a repetitive attractor, its own states should span
                fewer effective dimensions than the fixed trace's do.
  spectrum      the eigenvalue spectra behind that number.
  displacement  per prompt, the vector from its fixed-mean to its self-mean.
                Magnitude says how far; mean pairwise cosine says whether every
                prompt drifts the SAME way (a shared failure direction) or each
                drifts idiosyncratically.
  probe_scatter depth on x, cross-fitted probe score on y. Honest 2D: the y axis
                is supervised, but no point contributed to its own score.
  trajectory    per-prompt paths in that prompt's own 2D basis, fixed vs self,
                coloured by depth. The paired, dynamic view of the same thing.

Everything supervised here is cross-fitted with folds grouped by prompt, so no
point is ever scored by a model that saw its own problem.
"""
import argparse
import json
import os

import numpy as np


def load(npz, keys, enc, cond, L, w):
    k = f"{enc}|{cond}|L{L}|{w}"
    if k not in keys:
        return None
    return (np.asarray(npz[k], np.float32), np.asarray(npz[k + "|g"], int),
            np.asarray(npz[k + "|d"], int) if k + "|d" in npz else None)


def cross_fit_scores(Xf, gf, Xs, gs, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.concatenate([Xf, Xs]).astype(np.float32)
    y = np.concatenate([np.zeros(len(Xf)), np.ones(len(Xs))])
    g = np.concatenate([gf, gs])
    n = min(5, len(set(gf.tolist())), len(set(gs.tolist())))
    if n < 2 or len(X) < 40:
        return None, float("nan")
    s = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits=n).split(X, y, g):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=seed))
        clf.fit(X[tr], y[tr])
        s[te] = clf.decision_function(X[te])
    return s, float(roc_auc_score(y, s))


def mmd2(A, B, seed=0, cap=800):
    """Biased RBF-MMD^2 with the median heuristic. Subsampled to bound the O(n^2)."""
    rng = np.random.default_rng(seed)
    if len(A) > cap:
        A = A[rng.choice(len(A), cap, replace=False)]
    if len(B) > cap:
        B = B[rng.choice(len(B), cap, replace=False)]
    Z = np.concatenate([A, B])
    d2 = np.maximum(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1), 0)
    med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    K = np.exp(-d2 / med)
    n, m = len(A), len(B)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())


def mmd_null(A, gA, B, gB, reps=20, seed=0):
    """Paired permutation null: flip a prompt's two labels together, or not.

    Shuffling rows independently would break the group structure and give an
    optimistically tight band, so whole prompts are swapped instead.
    """
    rng = np.random.default_rng(seed)
    vals = []
    prompts = np.unique(np.concatenate([gA, gB]))
    for r in range(reps):
        flip = set(prompts[rng.random(len(prompts)) < 0.5].tolist())
        fa = np.array([g in flip for g in gA])
        fb = np.array([g in flip for g in gB])
        A2 = np.concatenate([A[~fa], B[fb]])
        B2 = np.concatenate([B[~fb], A[fa]])
        if len(A2) < 10 or len(B2) < 10:
            continue
        vals.append(mmd2(A2, B2, seed=r))
    return (float(np.percentile(vals, 95)) if vals else float("nan"))


def per_prompt_center(X, g):
    Xc = X.copy()
    for p in np.unique(g):
        m = g == p
        Xc[m] -= Xc[m].mean(0, keepdims=True)
    return Xc


def participation_ratio(X):
    """(sum lambda)^2 / sum lambda^2 -- effective number of dimensions spanned."""
    if len(X) < 4:
        return float("nan")
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def spectrum(X, k=40):
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    lam = lam / lam.sum()
    return lam[:k]


def depth_bins(d, n_bins):
    lo, hi = d.min(), d.max() + 1
    edges = np.linspace(lo, hi, n_bins + 1)
    return edges, np.clip(np.digitize(d, edges) - 1, 0, n_bins - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", required=True)
    ap.add_argument("--variant", required=True,
                    choices=["depth_auc", "depth_mmd", "effrank", "spectrum",
                             "displacement", "probe_scatter", "trajectory",
                             "cross_model", "cross_model_map"])
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--window", default="grid")
    ap.add_argument("--n_bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    out = args.outdir or os.path.join(args.states_dir, "visuals")
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(os.path.join(args.states_dir, "states_meta.json")))
    npz = np.load(os.path.join(args.states_dir, "states.npz"))
    keys = set(meta["keys"])
    labels = meta["labels"]
    L, w = args.layer, args.window
    tag = f"{args.variant}_L{L}_{w}"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {"dense": "#444444", "s50": "#2c7fb8", "s60": "#41ab5d", "s70": "#d95f0e"}
    C_FIX, C_SELF = "#2c7fb8", "#d95f0e"
    stats = {}

    def get(enc, cond):
        return load(npz, keys, enc, cond, L, w)

    def aligned(cond, lab):
        """Same tokens through two models, rows matched by (prompt, depth).

        This is the comparison with no text confound left in it: dense and the
        pruned model read an identical sequence, so any difference between the
        two state vectors is the pruning and nothing else. Rows are matched on
        the key rather than assumed to be in the same order.
        """
        A, B = get("dense", cond), get(lab, cond)
        if A is None or B is None or A[2] is None or B[2] is None:
            return None
        ia = {(int(g), int(d)): i for i, (g, d) in enumerate(zip(A[1], A[2]))}
        rows = [(ia[k], j) for j, k in
                ((j, (int(g), int(d))) for j, (g, d) in enumerate(zip(B[1], B[2])))
                if k in ia]
        if not rows:
            return None
        ra, rb = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        return A[0][ra], B[0][rb], B[1][rb], B[2][rb]

    # -------------------------------------------------------------- cross_model
    # How far pruning moves the representation, measured on identical tokens --
    # once under a fixed trace, once under the model's own rollout. If pruning
    # damages the state more in the regime the model actually generates in, the
    # self curve sits above the fixed curve and the gap widens with sparsity.
    if args.variant == "cross_model":
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for lab in labels:
            if lab == "dense":
                continue
            for cond, style, aname in ((("fixed"), "--", "fixed CoT"),
                                       ((f"self:{lab}"), "-", "self rollout")):
                al = aligned(cond, lab)
                if al is None:
                    continue
                Hd, Hp, g, d = al
                edges, _ = depth_bins(d, args.n_bins)
                xs, rel, cos = [], [], []
                for b in range(args.n_bins):
                    mm = (d >= edges[b]) & (d < edges[b + 1])
                    if mm.sum() < 20:
                        continue
                    a, p = Hd[mm], Hp[mm]
                    xs.append(0.5 * (edges[b] + edges[b + 1]))
                    rel.append(float(np.mean(np.linalg.norm(a - p, axis=1) /
                                             np.maximum(np.linalg.norm(a, axis=1), 1e-6))))
                    ca = (a * p).sum(1) / np.maximum(
                        np.linalg.norm(a, axis=1) * np.linalg.norm(p, axis=1), 1e-9)
                    cos.append(float(np.mean(1.0 - ca)))
                if not xs:
                    continue
                axes[0].plot(xs, rel, style, color=C.get(lab), lw=1.8,
                             label=f"{lab} {aname}")
                axes[1].plot(xs, cos, style, color=C.get(lab), lw=1.8,
                             label=f"{lab} {aname}")
                stats[f"{lab}|{cond}"] = {"depth": xs, "rel_l2": rel, "cos_dist": cos}
        axes[0].set_ylabel("‖h_dense − h_pruned‖ / ‖h_dense‖")
        axes[1].set_ylabel("1 − cos(h_dense, h_pruned)")
        for ax in axes:
            ax.set_xlabel("token depth into the continuation")
            ax.legend(fontsize=8)
        fig.suptitle(f"how far pruning moves the state, on IDENTICAL tokens — layer {L}\n"
                     "dashed = under a fixed trace, solid = under the model's own rollout",
                     fontsize=12)
        fig.tight_layout()

    # ---------------------------------------------------------- cross_model_map
    elif args.variant == "cross_model_map":
        from sklearn.decomposition import PCA
        conds = ["fixed", "self"]
        rows = [l for l in labels if l != "dense"]
        fig, axes = plt.subplots(len(rows), 2, figsize=(9.5, 4.2 * len(rows)))
        axes = np.atleast_2d(axes)
        for r, lab in enumerate(rows):
            for c, cname in enumerate(conds):
                ax = axes[r][c]
                cond = "fixed" if cname == "fixed" else f"self:{lab}"
                al = aligned(cond, lab)
                if al is None:
                    ax.axis("off"); continue
                Hd, Hp, g, d = al
                # one shared basis per panel, fit on the dense states, so the
                # pruned cloud is shown as a displacement from the reference
                pca = PCA(n_components=2, random_state=args.seed).fit(Hd)
                Zd, Zp = pca.transform(Hd), pca.transform(Hp)
                k = min(600, len(Zd))
                sel = np.random.default_rng(args.seed).choice(len(Zd), k, replace=False)
                ax.scatter(Zd[sel, 0], Zd[sel, 1], s=5, alpha=.4, c="#444444",
                           linewidths=0, label="dense")
                ax.scatter(Zp[sel, 0], Zp[sel, 1], s=5, alpha=.4, c=C.get(lab),
                           linewidths=0, label=lab)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title("fixed CoT" if c == 0 else "self rollout")
                if c == 0:
                    ax.set_ylabel(lab)
                if r == 0 and c == 0:
                    ax.legend(markerscale=3, fontsize=8)
        fig.suptitle(f"dense vs pruned states on identical tokens — layer {L}\n"
                     "basis fit on the dense cloud in each panel", fontsize=12)
        fig.tight_layout()

    # ---------------------------------------------------------------- depth_auc
    elif args.variant in ("depth_auc", "depth_mmd"):
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for lab in labels:
            F, S = get(lab, "fixed"), get(lab, f"self:{lab}")
            if F is None or S is None or F[2] is None:
                continue
            Xf, gf, df = F
            Xs, gs, ds = S
            edges, _ = depth_bins(np.concatenate([df, ds]), args.n_bins)
            xs, ys, nulls = [], [], []
            for b in range(args.n_bins):
                mf = (df >= edges[b]) & (df < edges[b + 1])
                ms = (ds >= edges[b]) & (ds < edges[b + 1])
                if mf.sum() < 30 or ms.sum() < 30:
                    continue
                xs.append(0.5 * (edges[b] + edges[b + 1]))
                if args.variant == "depth_auc":
                    ys.append(cross_fit_scores(Xf[mf], gf[mf], Xs[ms], gs[ms], args.seed)[1])
                else:
                    ys.append(mmd2(Xf[mf], Xs[ms], args.seed))
                    nulls.append(mmd_null(Xf[mf], gf[mf], Xs[ms], gs[ms], seed=args.seed))
            if not xs:
                continue
            ax.plot(xs, ys, "-o", color=C.get(lab, None), label=lab, ms=4)
            stats[lab] = {"depth": xs, "y": ys}
            if nulls:
                ax.plot(xs, nulls, ":", color=C.get(lab, None), alpha=.5, lw=1)
            # dense-encoder control on the same tokens
            if lab != "dense" and args.variant == "depth_auc":
                DF, DS = get("dense", "fixed"), get("dense", f"self:{lab}")
                if DF is not None and DS is not None:
                    Xdf, gdf, ddf = DF
                    Xds, gds, dds = DS
                    ys2, xs2 = [], []
                    for b in range(args.n_bins):
                        mf = (ddf >= edges[b]) & (ddf < edges[b + 1])
                        ms = (dds >= edges[b]) & (dds < edges[b + 1])
                        if mf.sum() < 30 or ms.sum() < 30:
                            continue
                        xs2.append(0.5 * (edges[b] + edges[b + 1]))
                        ys2.append(cross_fit_scores(Xdf[mf], gdf[mf], Xds[ms],
                                                    gds[ms], args.seed)[1])
                    if xs2:
                        ax.plot(xs2, ys2, "--", color=C.get(lab, None), alpha=.55, lw=1.4)
                        stats[lab + "_dense_encoder"] = {"depth": xs2, "y": ys2}
        ax.set_xlabel("token depth into the continuation")
        if args.variant == "depth_auc":
            ax.set_ylabel("cross-fitted probe AUC  (fixed vs self)")
            ax.axhline(0.5, color="k", lw=.8, ls=":")
            ax.set_title(f"separability vs depth — layer {L}\n"
                         "solid = own encoder, dashed = dense encoder on the same tokens",
                         fontsize=11)
        else:
            ax.set_ylabel("RBF-MMD$^2$  (dotted = 95th pct paired permutation null)")
            ax.set_title(f"distribution distance vs depth — layer {L}", fontsize=11)
        ax.legend()
        fig.tight_layout()

    # ------------------------------------------------------------------ effrank
    elif args.variant == "effrank":
        fig, ax = plt.subplots(figsize=(7.5, 5))
        xs = np.arange(len(labels))
        pf, ps = [], []
        for lab in labels:
            F, S = get(lab, "fixed"), get(lab, f"self:{lab}")
            # per-prompt centering first: otherwise this measures how varied the
            # problems are, which is identical across panels by construction
            pf.append(participation_ratio(per_prompt_center(F[0], F[1])) if F else np.nan)
            ps.append(participation_ratio(per_prompt_center(S[0], S[1])) if S else np.nan)
        ax.bar(xs - .2, pf, .4, label="fixed CoT", color=C_FIX)
        ax.bar(xs + .2, ps, .4, label="self rollout", color=C_SELF)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_ylabel("participation ratio (effective dimensions)")
        ax.set_title(f"how many dimensions each state cloud spans — layer {L}\n"
                     "prompt means removed first", fontsize=11)
        ax.legend()
        stats = {lab: {"fixed": float(a), "self": float(b)}
                 for lab, a, b in zip(labels, pf, ps)}
        fig.tight_layout()

    # ----------------------------------------------------------------- spectrum
    elif args.variant == "spectrum":
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for lab in labels:
            F, S = get(lab, "fixed"), get(lab, f"self:{lab}")
            if F is None or S is None:
                continue
            ax.semilogy(spectrum(per_prompt_center(F[0], F[1])), "-",
                        color=C.get(lab), alpha=.55, lw=1.2)
            ax.semilogy(spectrum(per_prompt_center(S[0], S[1])), "--",
                        color=C.get(lab), lw=1.8, label=f"{lab} self")
        ax.set_xlabel("component"); ax.set_ylabel("explained variance fraction")
        ax.set_title(f"state covariance spectra — layer {L}\n"
                     "solid = fixed CoT, dashed = self rollout", fontsize=11)
        ax.legend(fontsize=8)
        fig.tight_layout()

    # ------------------------------------------------------------- displacement
    elif args.variant == "displacement":
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        mags, coss = {}, {}
        for lab in labels:
            F, S = get(lab, "fixed"), get(lab, f"self:{lab}")
            if F is None or S is None:
                continue
            Xf, gf, _ = F
            Xs, gs, _ = S
            common = sorted(set(gf.tolist()) & set(gs.tolist()))
            if len(common) < 5:
                continue
            D = np.stack([Xs[gs == p].mean(0) - Xf[gf == p].mean(0) for p in common])
            # scale-free: how big is the shift next to the spread within a regime
            spread = np.mean([np.linalg.norm(Xf[gf == p] - Xf[gf == p].mean(0), axis=1).mean()
                              for p in common])
            mags[lab] = np.linalg.norm(D, axis=1) / max(spread, 1e-6)
            U = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)
            Cm = U @ U.T
            iu = np.triu_indices(len(U), 1)
            coss[lab] = Cm[iu]
        axes[0].boxplot([mags[l] for l in mags], labels=list(mags), showfliers=False)
        axes[0].set_ylabel("‖self mean − fixed mean‖ / within-regime spread")
        axes[0].set_title("how far each prompt drifts", fontsize=11)
        axes[1].boxplot([coss[l] for l in coss], labels=list(coss), showfliers=False)
        axes[1].axhline(0, color="k", lw=.8, ls=":")
        axes[1].set_ylabel("pairwise cosine between prompts' drift vectors")
        axes[1].set_title("do all prompts drift the SAME way?", fontsize=11)
        stats = {l: {"mag_median": float(np.median(mags[l])),
                     "cos_median": float(np.median(coss[l]))} for l in mags}
        fig.suptitle(f"per-prompt displacement — layer {L}", fontsize=12)
        fig.tight_layout()

    # ------------------------------------------------------------ probe_scatter
    elif args.variant == "probe_scatter":
        fig, axes = plt.subplots(1, len(labels), figsize=(4.3 * len(labels), 4.4),
                                 sharey=True)
        for ax, lab in zip(np.atleast_1d(axes), labels):
            F, S = get(lab, "fixed"), get(lab, f"self:{lab}")
            if F is None or S is None or F[2] is None:
                ax.axis("off"); continue
            Xf, gf, df = F
            Xs, gs, ds = S
            s, auc = cross_fit_scores(Xf, gf, Xs, gs, args.seed)
            stats[lab] = auc
            ax.scatter(df, s[:len(Xf)], s=3, alpha=.35, c=C_FIX, linewidths=0,
                       label="fixed CoT")
            ax.scatter(ds, s[len(Xf):], s=3, alpha=.35, c=C_SELF, linewidths=0,
                       label="self rollout")
            ax.axhline(0, color="k", lw=.8, ls="--")
            ax.set_title(f"{lab}   AUC {auc:.3f}")
            ax.set_xlabel("token depth")
        np.atleast_1d(axes)[0].set_ylabel("cross-fitted probe score")
        np.atleast_1d(axes)[0].legend(markerscale=4, fontsize=8)
        fig.suptitle(f"probe score vs depth — layer {L} (out-of-sample scores)",
                     fontsize=12)
        fig.tight_layout()

    # --------------------------------------------------------------- trajectory
    else:
        from sklearn.decomposition import PCA
        n_show = 3
        fig, axes = plt.subplots(n_show, len(labels),
                                 figsize=(3.9 * len(labels), 3.6 * n_show))
        for c, lab in enumerate(labels):
            F, S = get(lab, "fixed"), get(lab, f"self:{lab}")
            if F is None or S is None or F[2] is None:
                continue
            Xf, gf, df = F
            Xs, gs, ds = S
            common = sorted(set(gf.tolist()) & set(gs.tolist()))[:n_show]
            for r, p in enumerate(common):
                ax = axes[r][c] if n_show > 1 else axes[c]
                a, b = Xf[gf == p], Xs[gs == p]
                da, db = df[gf == p], ds[gs == p]
                oa, ob = np.argsort(da), np.argsort(db)
                a, b, da, db = a[oa], b[ob], da[oa], db[ob]
                # basis from this prompt alone, so the panel shows the paired
                # divergence rather than where this problem sits overall
                pca = PCA(n_components=2, random_state=args.seed).fit(
                    np.concatenate([a, b]))
                Za, Zb = pca.transform(a), pca.transform(b)
                ax.plot(Za[:, 0], Za[:, 1], "-", color=C_FIX, alpha=.5, lw=1)
                ax.plot(Zb[:, 0], Zb[:, 1], "-", color=C_SELF, alpha=.5, lw=1)
                ax.scatter(Za[:, 0], Za[:, 1], c=da, cmap="Blues", s=18, zorder=3)
                ax.scatter(Zb[:, 0], Zb[:, 1], c=db, cmap="Oranges", s=18, zorder=3)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(lab)
                if c == 0:
                    ax.set_ylabel(f"prompt {p}")
        fig.suptitle(f"per-prompt trajectories, layer {L} — blue = fixed CoT, "
                     "orange = self rollout (shade = depth)", fontsize=12)
        fig.tight_layout()

    p = os.path.join(out, f"{tag}.png")
    fig.savefig(p, dpi=150)
    if stats:
        json.dump(stats, open(os.path.join(out, f"{tag}.json"), "w"), indent=2,
                  default=float)
        print(f"[vis] {tag} stats -> {tag}.json", flush=True)
    print(f"[vis] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
