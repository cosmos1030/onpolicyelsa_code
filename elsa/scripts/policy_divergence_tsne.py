#!/usr/bin/env python
"""Where each model's rollouts land, for ONE prompt at a time.

The earlier t-SNE attempts failed for two reasons that this design removes by
construction rather than by correction:

  prompt identity   Sampling many problems made "which problem is this" the
                    dominant neighbourhood structure, and the regime label had
                    no room left in 2D. Here the prompt is HELD FIXED, so it is
                    not a variable at all.
  correlated points States taken from one sequence share nearly all their
                    context, so 32 of them are worth far less than 32 samples.
                    Here each point is a whole rollout -- one prompt sampled K
                    times per model, so the points are independent draws from
                    the thing actually being compared: the model's own output
                    distribution.

Every rollout is encoded by the DENSE model alone. One coordinate system for
all of them means the map shows where the TEXT sits, not how each model happens
to represent its own text -- which is exactly the claim this figure makes.

What it can show: the distribution of trajectories moves away from the dense
model's as pruning gets harder. What it cannot show: that the move costs
anything, or that on-policy training repairs it. Those are separate questions,
and the KL diagnostics answered them the other way.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onpolicy_mismatch_diag import build_prompts  # noqa: E402


def sample_many(model_path, prompts, n_samples, max_new, temperature, gpu_mem,
                seed, max_model_len):
    """K rollouts per prompt, in one vLLM request per prompt."""
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_path, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=gpu_mem, max_model_len=max_model_len,
              enforce_eager=False, seed=seed)
    sp = SamplingParams(n=n_samples, temperature=temperature, top_p=0.95,
                        top_k=20, max_tokens=max_new)
    outs = llm.generate(prompts, sp)
    # [prompt][sample] -> token ids
    gens = [[list(o.token_ids) for o in out.outputs] for out in outs]
    del llm
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return gens


@torch.no_grad()
def pool_states(model, tokenizer, prompt, rollouts, layers, windows, device):
    """One vector per rollout per (layer, window): the mean state over that span.

    Pooling to a single vector is the point -- it makes each rollout one
    independent sample instead of a correlated cloud of its own.
    """
    need = max(hi for _, hi in windows.values())
    p_ids = tokenizer(prompt, return_tensors="pt",
                      add_special_tokens=False).input_ids[0]
    n_prompt = p_ids.numel()
    out = {L: {w: [] for w in windows} for L in layers}
    kept = {w: [] for w in windows}
    for ri, gen in enumerate(rollouts):
        if not gen:
            continue
        c_ids = torch.tensor(list(gen[:need]), dtype=p_ids.dtype)
        ids = torch.cat([p_ids, c_ids]).unsqueeze(0).to(device)
        hs = model.model(ids, output_hidden_states=True).hidden_states
        for wname, (lo, hi) in windows.items():
            a, b = n_prompt + lo, n_prompt + min(hi, len(c_ids))
            if b - a < 16:
                continue        # this rollout never reached the window
            kept[wname].append(ri)
            for L in layers:
                out[L][wname].append(
                    hs[L][0, a:b].float().mean(0).cpu().numpy().astype(np.float32))
        del hs
        torch.cuda.empty_cache()
    packed = {L: {} for L in layers}
    for L in layers:
        for w in windows:
            v = out[L][w]
            packed[L][w] = np.stack(v) if v else np.zeros((0, 1), np.float32)
    return packed, {w: len(kept[w]) for w in windows}


def separability(A, B, seed=0):
    """Cross-validated AUC between two clouds of independent samples."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(A) < 10 or len(B) < 10:
        return float("nan")
    X = np.concatenate([A, B]).astype(np.float32)
    y = np.concatenate([np.zeros(len(A)), np.ones(len(B))])
    s = np.zeros(len(X))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=seed))
        clf.fit(X[tr], y[tr])
        s[te] = clf.decision_function(X[te])
    return float(roc_auc_score(y, s))


def mmd2(A, B, seed=0):
    rng = np.random.default_rng(seed)
    Z = np.concatenate([A, B]).astype(np.float32)
    sq = (Z * Z).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    med = np.median(d2[d2 > 0]) if (d2 > 0).any() else 1.0
    K = np.exp(-d2 / med)
    n = len(A)
    return float(K[:n, :n].mean() + K[n:, n:].mean() - 2 * K[:n, n:].mean())


def embed(X, seed=0, perplexity=30):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    if len(X) < 10:
        return None
    d = min(50, X.shape[1], len(X) - 1)
    Xp = PCA(n_components=d, random_state=seed).fit_transform(X.astype(np.float32))
    p = min(perplexity, max(5, (len(Xp) - 1) // 4))
    return TSNE(n_components=2, perplexity=p, init="pca", random_state=seed,
                max_iter=1000).fit_transform(Xp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense_model", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="family:label=path, e.g. alps:s50=cosmos1030/...")
    ap.add_argument("--n_prompts", type=int, default=6)
    ap.add_argument("--n_samples", type=int, default=96)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 36])
    ap.add_argument("--prompt_source", default="ot3", choices=["ot3", "math500"])
    # Held-out by index is not held-out by content: OpenThoughts3 repeats
    # problems across indices, so a prompt from the evaluation window can be a
    # duplicate of one in the calibration pool. Point this at the training
    # JSONL and such prompts are rejected at selection time.
    ap.add_argument("--dedup_against",
                    default="/home1/doyoonkim/projects/elsa/data/"
                            "ot3_fineweb_40k_qwen3_nostrip_8192.jsonl",
                    help="training JSONL to exclude prompts against ('' disables)")
    ap.add_argument("--with_teacher", action="store_true",
                    help="also embed the dataset's own CoT as a landmark point")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    # Validate before anything expensive: a stray shell argument in this list
    # would otherwise surface as an unnamed unpack error half an hour in,
    # after the tokenizer and dataset have already loaded.
    for m in args.models:
        if "=" not in m or ":" not in m.split("=", 1)[0]:
            raise SystemExit(
                f"--models entries must look like family:label=path; got {m!r}")

    os.makedirs(args.outdir, exist_ok=True)
    windows = {"early": (0, 512), "late": (1024, 2048)}

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.dense_model, trust_remote_code=True)
    # build_prompts pulls all of OpenThoughts3-1.2M (120 files, ~25 min) and
    # parses 1.2M rows to hand back six prompts. Cache the six.
    pcache = os.path.join(args.outdir, "prompts.json")
    # dedup_against is part of the cache key: a set built without the training
    # filter is a DIFFERENT set of prompts, and silently reusing it would put
    # contaminated prompts back into a run that asked for clean ones.
    _want = {"source": args.prompt_source, "seed": args.seed,
             "n_prompts": args.n_prompts,
             "dedup_against": args.dedup_against or None}
    _c = json.load(open(pcache)) if os.path.exists(pcache) else None
    # A cache keyed only on the path would silently serve the wrong prompts
    # after a change of seed or count, and the rollouts cached beside it are
    # keyed on nothing at all -- so refuse rather than mix them.
    if _c and all(_c.get(k) == v for k, v in _want.items()):
        prompts, solutions, pids = _c["prompts"], _c["solutions"], _c["ids"]
        print(f"[pol] {len(prompts)} prompts from cache", flush=True)
    else:
        if _c:
            raise SystemExit(
                f"prompt cache in {args.outdir} was built for "
                f"{ {k: _c.get(k) for k in _want} } but this run wants {_want}. "
                "The cached rollouts belong to those prompts too -- use a "
                "different --outdir rather than mixing them.")
        prompts, solutions, pids = build_prompts(
            tok, args.n_prompts, args.seed, True, args.prompt_source,
            train_file=args.dedup_against or None)
        tmp = pcache + ".tmp"
        json.dump({"prompts": prompts, "solutions": solutions, "ids": pids,
                   "source": args.prompt_source, "seed": args.seed,
                   "n_prompts": args.n_prompts,
                   "dedup_against": args.dedup_against or None}, open(tmp, "w"))
        os.replace(tmp, pcache)
    plens = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    mml = max(plens) + args.max_new_tokens + 64
    print(f"[pol] {len(prompts)} prompts, {args.n_samples} samples each, "
          f"max_model_len {mml}", flush=True)

    # Key on the full "family:label" -- both families carry an s50, and keying
    # on the bare label silently drops one of them.
    specs = {"dense": ("dense", "dense", args.dense_model)}
    for m in args.models:
        # Fail here with the offending token rather than inside an unpack: a
        # stray shell argument landing in this list is easy to do and the
        # ValueError it causes names nothing.
        fam_lab, path = m.split("=", 1)
        fam, lab = fam_lab.split(":", 1)
        specs[f"{fam}:{lab}"] = (fam, lab, path)

    # -- sample, cached per model
    rollouts = {}
    for key, (fam, lab, path) in specs.items():
        cache = os.path.join(args.outdir, f"samples_{key.replace(':', '_')}.json")
        if os.path.exists(cache):
            rollouts[key] = json.load(open(cache))
            print(f"[pol] {key}: cached", flush=True)
            continue
        print(f"[pol] sampling {key} ...", flush=True)
        g = sample_many(path, prompts, args.n_samples, args.max_new_tokens,
                        args.temperature, args.gpu_mem, args.seed, mml)
        json.dump(g, open(cache, "w"))
        rollouts[key] = g
        lens = [len(s) for pr in g for s in pr]
        print(f"[pol] {key}: mean len {np.mean(lens):.0f}, "
              f"cap-hit {np.mean([l >= args.max_new_tokens for l in lens]):.2f}",
              flush=True)

    # -- encode everything with the dense model: one coordinate system
    print("[pol] encoding all rollouts with the dense model ...", flush=True)
    dense = AutoModelForCausalLM.from_pretrained(
        args.dense_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).to("cuda").eval()
    # states[prompt_idx][label][layer][window] -> (n_samples, hidden)
    states = [{} for _ in prompts]
    for pi, prompt in enumerate(prompts):
        for key in specs:
            packed, cov = pool_states(dense, tok, prompt, rollouts[key][pi],
                                      args.layers, windows, "cuda")
            states[pi][key] = packed
        # The dataset's own CoT, through the same encoder and the same windows.
        # OT3 ships exactly one trace per problem, so this is a single landmark
        # point per panel rather than a cloud -- it cannot carry an MMD, but it
        # shows WHICH WAY the models sit relative to the text they were trained
        # on, which "distance from dense" alone does not say.
        if args.with_teacher:
            t_ids = tok(solutions[pi], add_special_tokens=False).input_ids
            tp, _ = pool_states(dense, tok, prompt, [t_ids], args.layers,
                                windows, "cuda")
            states[pi]["teacher"] = tp
        print(f"[pol]   prompt {pi}: done ({cov})", flush=True)
    del dense
    torch.cuda.empty_cache()

    # -- numbers
    labs = list(specs)
    results = {}
    print("\n[pol] === divergence from dense (AUC / MMD), per prompt ===", flush=True)
    for L in args.layers:
        for w in windows:
            print(f"  layer {L}, window {w}:", flush=True)
            for lab in labs[1:]:
                aucs, mmds = [], []
                for pi in range(len(prompts)):
                    A = states[pi]["dense"][L][w]
                    B = states[pi][lab][L][w]
                    if len(A) < 10 or len(B) < 10:
                        continue
                    aucs.append(separability(A, B, args.seed))
                    mmds.append(mmd2(A, B, args.seed))
                if not aucs:
                    continue
                results[f"L{L}/{w}/{lab}"] = {
                    "auc_mean": float(np.mean(aucs)), "auc_per_prompt": aucs,
                    "mmd_mean": float(np.mean(mmds)), "mmd_per_prompt": mmds}
                print(f"    {lab:<14} AUC {np.mean(aucs):.3f}   "
                      f"MMD {np.mean(mmds):.4f}  "
                      f"(per-prompt MMD {' '.join(f'{m:.3f}' for m in mmds)})", flush=True)
    if args.with_teacher:
        print("\n[pol] === distance from each cloud's centre to the dataset CoT ===",
              flush=True)
        print("      (in units of the dense cloud's own spread: 1.0 means the "
              "teacher sits as far from that model as the model's rollouts "
              "scatter)", flush=True)
        for L in args.layers:
            for w in windows:
                print(f"  layer {L}, {w}:", flush=True)
                for lab in labs:
                    ds = []
                    for pi in range(len(prompts)):
                        X = states[pi][lab][L][w]
                        T = states[pi].get("teacher", {}).get(L, {}).get(w)
                        D = states[pi]["dense"][L][w]
                        if T is None or len(T) == 0 or len(X) < 5 or len(D) < 5:
                            continue
                        spread = np.linalg.norm(D - D.mean(0), axis=1).mean()
                        ds.append(float(np.linalg.norm(X.mean(0) - T[0]) /
                                        max(spread, 1e-6)))
                    if ds:
                        results[f"L{L}/{w}/{lab}/to_teacher"] = ds
                        print(f"    {lab:<14} {np.mean(ds):6.3f}   "
                              f"(per prompt {' '.join(f'{d:.2f}' for d in ds)})",
                              flush=True)

    json.dump(results, open(os.path.join(args.outdir, "divergence.json"), "w"),
              indent=2)

    # Encoding is the expensive step and the figures are the part worth
    # iterating on, so keep the pooled vectors. Written then renamed: the
    # analysis side reads this path.
    blob = {}
    for pi in range(len(prompts)):
        for k in list(specs) + (["teacher"] if args.with_teacher else []):
            for L in args.layers:
                for w in windows:
                    v = states[pi].get(k, {}).get(L, {}).get(w)
                    if v is not None and len(v):
                        blob[f"p{pi}|{k}|L{L}|{w}"] = v
    tmp = os.path.join(args.outdir, "pooled.npz.tmp.npz")
    np.savez(tmp, **blob)
    os.replace(tmp, os.path.join(args.outdir, "pooled.npz"))
    json.dump({"prompts": len(prompts), "specs": {k: list(v) for k, v in specs.items()},
               "layers": args.layers, "windows": {k: list(v) for k, v in windows.items()}},
              open(os.path.join(args.outdir, "pooled_meta.json"), "w"), indent=2)
    print(f"[pol] wrote pooled.npz ({len(blob)} arrays)", flush=True)

    # -- figures, one row per family
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = sorted({f for f, _, _ in specs.values()} - {"dense"})
    # by-sparsity figures colour the sparsity; by-sparsity-level figures colour
    # the METHOD, since that is what those panels contrast.
    palette = {"dense": "#3f4652", "s30": "#7fb2d4", "s40": "#4a93c4",
               "s50": "#2c7fb8", "s60": "#41a05d", "s70": "#d95f0e"}
    fam_palette = {"dense": "#3f4652", "alps": "#d95f0e", "sparsegpt": "#8c4a9e",
                   "ours": "#0f8b7e", "alps_sft": "#b08600"}
    for L in args.layers:
        for w in windows:
            for fam in fams:
                members = ["dense"] + sorted(
                    k for k, (f, _, _) in specs.items() if f == fam)
                ncol = min(len(prompts), 6)
                fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 4.0))
                axes = np.atleast_1d(axes)
                for pi in range(ncol):
                    ax = axes[pi]
                    Xs, labels = [], []
                    for lab in members:
                        v = states[pi][lab][L][w]
                        if len(v) < 5:
                            continue
                        Xs.append(v)
                        labels += [lab] * len(v)
                    if not Xs:
                        ax.axis("off"); continue
                    Z = embed(np.concatenate(Xs), args.seed)
                    if Z is None:
                        ax.axis("off"); continue
                    labels = np.array(labels)
                    for lab in members:
                        m = labels == lab
                        if not m.any():
                            continue
                        disp = specs[lab][1]
                        ax.scatter(Z[m, 0], Z[m, 1], s=13, alpha=.72,
                                   c=palette.get(disp, "#888"), linewidths=0,
                                   label=disp)
                    ax.set_xticks([]); ax.set_yticks([])
                    # MMD, not AUC: AUC saturates at 1.00 for every model at
                    # every sparsity -- rollouts from different models are
                    # trivially separable and the number says nothing about
                    # HOW far apart they are.
                    sub = "  ".join(
                        f"{specs[lab][1]} {results.get(f'L{L}/{w}/{lab}', {}).get('mmd_per_prompt', [float('nan')] * 99)[pi]:.3f}"
                        for lab in members[1:])
                    ax.set_title(f"prompt {pi}\nMMD$^2$ to dense: {sub}", fontsize=8)
                axes[0].legend(loc="upper left", markerscale=1.6, fontsize=8.5,
                               framealpha=.9)
                fig.suptitle(
                    f"{fam.upper()} — one prompt per panel, {args.n_samples} rollouts per model "
                    f"(layer {L}, mean over continuation tokens {windows[w][0]}-{windows[w][1]}, "
                    f"dense encoder)", fontsize=12)
                fig.tight_layout(rect=[0, 0, 1, 0.92])
                p = os.path.join(args.outdir, f"tsne_{fam}_L{L}_{w}.png")
                fig.savefig(p, dpi=150)
                plt.close(fig)
                print(f"[pol] wrote {p}", flush=True)

            # --- the comparison that carries the claim: at ONE sparsity, where
            # each method's rollouts land relative to dense. A baseline drifting
            # away only means something next to a method that does not.
            levels = sorted({lab for _, lab, _ in specs.values()} - {"dense"})
            for lev in levels:
                members = ["dense"] + sorted(
                    k for k, (f, l, _) in specs.items() if l == lev)
                if args.with_teacher and "teacher" in states[0]:
                    members = members + ["teacher"]
                if len(members) < 3:
                    continue          # nothing to contrast at this sparsity
                ncol = min(len(prompts), 6)
                fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 4.0))
                axes = np.atleast_1d(axes)
                for pi in range(ncol):
                    ax = axes[pi]
                    Xs, labels = [], []
                    for k in members:
                        v = states[pi].get(k, {}).get(L, {}).get(w)
                        # teacher is a single landmark row -- the >=5 floor that
                        # guards the clouds would silently drop it
                        if v is None or len(v) < (1 if k == "teacher" else 5):
                            continue
                        Xs.append(v)
                        labels += [k] * len(v)
                    if not Xs:
                        ax.axis("off"); continue
                    Z = embed(np.concatenate(Xs), args.seed)
                    if Z is None:
                        ax.axis("off"); continue
                    labels = np.array(labels)
                    for k in members:
                        m = labels == k
                        if not m.any():
                            continue
                        if k == "teacher":
                            ax.scatter(Z[m, 0], Z[m, 1], s=260, marker="*",
                                       c="#c9184a", edgecolors="white",
                                       linewidths=1.2, zorder=5,
                                       label="dataset CoT")
                        else:
                            ax.scatter(Z[m, 0], Z[m, 1], s=13, alpha=.72,
                                       c=fam_palette.get(specs[k][0], "#888"),
                                       linewidths=0, label=specs[k][0])
                    ax.set_xticks([]); ax.set_yticks([])
                    sub = "\n".join(
                        f"{specs[k][0]:<10}{results.get(f'L{L}/{w}/{k}', {}).get('mmd_per_prompt', [float('nan')] * 99)[pi]:.3f}"
                        for k in members[1:] if k != "teacher")
                    ax.set_title(f"prompt {pi}\nMMD$^2$ to dense\n{sub}",
                                 fontsize=7.5, fontfamily="monospace", loc="left")
                axes[0].legend(loc="upper left", markerscale=1.6, fontsize=8.5,
                               framealpha=.9)
                fig.suptitle(
                    f"{lev.upper()} — methods compared at one sparsity. One prompt per panel, "
                    f"{args.n_samples} rollouts per model (layer {L}, mean over continuation "
                    f"tokens {windows[w][0]}-{windows[w][1]}, dense encoder)", fontsize=11)
                fig.tight_layout(rect=[0, 0, 1, 0.90])
                p = os.path.join(args.outdir, f"tsne_bylevel_{lev}_L{L}_{w}.png")
                fig.savefig(p, dpi=150)
                plt.close(fig)
                print(f"[pol] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
