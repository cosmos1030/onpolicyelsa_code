#!/usr/bin/env python
"""The dataset's CoT, read by every model — the other axis.

policy_divergence_tsne.py holds the ENCODER fixed (dense) and varies the text,
so it asks where each model's generated text lands. That map necessarily shows
the dataset CoT as a single point: one text, one encoding.

This asks the mirror question. The text is held fixed -- the dataset's own CoT --
and it is read by each model in turn, so every model contributes its own point.
What moves now is the representation, not the text.

Two things make the overlay legitimate:

  aligned bases   ALPS, SparseGPT and our method all mask and update the dense
                  weights rather than retraining from scratch, so neuron i still
                  means what it meant in dense. Encodings from different models
                  can share one map. This would be meaningless for two
                  independently trained models.
  a scale bar     A displacement is only interesting next to something. The
                  dense model's own rollout cloud is plotted alongside, so the
                  question becomes: does pruning move the CoT further than
                  dense's own generations scatter?

The CoT is also cut into fixed-length segments, each pooled separately, so each
model contributes a small cloud instead of a lone point. Those segments are not
independent samples -- they are pieces of one text -- so read their spread as
the extent of the CoT in representation space, not as a sampling distribution.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@torch.no_grad()
def cot_segments(model, tokenizer, prompt, cot_ids, layers, seg_len, n_seg, device):
    """Pooled state of each segment of one text. Returns {layer: (n_seg, hidden)}."""
    p_ids = tokenizer(prompt, return_tensors="pt",
                      add_special_tokens=False).input_ids[0]
    need = seg_len * n_seg
    c_ids = torch.tensor(list(cot_ids[:need]), dtype=p_ids.dtype)
    ids = torch.cat([p_ids, c_ids]).unsqueeze(0).to(device)
    n_prompt = p_ids.numel()
    hs = model.model(ids, output_hidden_states=True).hidden_states
    out = {}
    for L in layers:
        h = hs[L][0]
        rows = []
        for k in range(n_seg):
            a, b = n_prompt + k * seg_len, n_prompt + min((k + 1) * seg_len, len(c_ids))
            if b - a < 16:
                break
            rows.append(h[a:b].float().mean(0).cpu().numpy().astype(np.float32))
        out[L] = np.stack(rows) if rows else np.zeros((0, 1), np.float32)
    del hs
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def rollout_cloud(model, tokenizer, prompt, rollouts, layers, lo, hi, device, cap):
    """Dense's own rollouts, pooled the same way -- the scale bar."""
    p_ids = tokenizer(prompt, return_tensors="pt",
                      add_special_tokens=False).input_ids[0]
    n_prompt = p_ids.numel()
    out = {L: [] for L in layers}
    for gen in rollouts[:cap]:
        if len(gen) < hi:
            continue
        ids = torch.cat([p_ids, torch.tensor(list(gen[:hi]), dtype=p_ids.dtype)]
                        ).unsqueeze(0).to(device)
        hs = model.model(ids, output_hidden_states=True).hidden_states
        for L in layers:
            out[L].append(hs[L][0, n_prompt + lo:n_prompt + hi].float()
                          .mean(0).cpu().numpy().astype(np.float32))
        del hs
        torch.cuda.empty_cache()
    return {L: (np.stack(v) if v else np.zeros((0, 1), np.float32))
            for L, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True,
                    help="the policy-divergence outdir: reuses its prompts.json "
                         "and samples_dense.json")
    ap.add_argument("--models", nargs="+", required=True, help="label=path")
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 36])
    ap.add_argument("--seg_len", type=int, default=256)
    ap.add_argument("--n_seg", type=int, default=8)
    ap.add_argument("--scale_cap", type=int, default=48,
                    help="dense rollouts used for the scale bar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    pc = json.load(open(os.path.join(args.samples_dir, "prompts.json")))
    prompts, solutions = pc["prompts"], pc["solutions"]
    dense_rolls = json.load(open(os.path.join(args.samples_dir, "samples_dense.json")))
    print(f"[cot] {len(prompts)} prompts from cache", flush=True)

    specs = {}
    for m in args.models:
        lab, path = m.split("=", 1)
        specs[lab] = path
    tok = AutoTokenizer.from_pretrained(specs["dense"], trust_remote_code=True)
    cot_ids = [tok(s, add_special_tokens=False).input_ids for s in solutions]
    print(f"[cot] CoT token lengths: {[len(c) for c in cot_ids]}", flush=True)

    # states[prompt][label][layer] -> (n_seg, hidden);  scale[prompt][layer]
    states = [{} for _ in prompts]
    scale = [{} for _ in prompts]
    for lab, path in specs.items():
        print(f"[cot] encoding with {lab} ...", flush=True)
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
        for pi, prompt in enumerate(prompts):
            states[pi][lab] = cot_segments(m, tok, prompt, cot_ids[pi], args.layers,
                                           args.seg_len, args.n_seg, "cuda")
            if lab == "dense":
                scale[pi] = rollout_cloud(m, tok, prompt, dense_rolls[pi],
                                          args.layers, 0, args.seg_len * args.n_seg,
                                          "cuda", args.scale_cap)
        del m
        torch.cuda.empty_cache()

    labs = list(specs)
    results = {}
    print("\n[cot] === how far each model moves the SAME text ===", flush=True)
    print("      (‖h_model − h_dense‖ on the CoT, in units of the dense rollout "
          "cloud's own spread)", flush=True)
    for L in args.layers:
        print(f"  layer {L}:", flush=True)
        for lab in labs:
            if lab == "dense":
                continue
            ds = []
            for pi in range(len(prompts)):
                A, B = states[pi]["dense"][L], states[pi][lab][L]
                S = scale[pi].get(L)
                if len(A) == 0 or len(B) == 0 or S is None or len(S) < 4:
                    continue
                n = min(len(A), len(B))
                spread = np.linalg.norm(S - S.mean(0), axis=1).mean()
                ds.append(float(np.linalg.norm(A[:n] - B[:n], axis=1).mean() /
                                max(spread, 1e-6)))
            if ds:
                results[f"L{L}/{lab}"] = ds
                print(f"    {lab:<14} {np.mean(ds):6.3f}   "
                      f"(per prompt {' '.join(f'{d:.2f}' for d in ds)})", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    C = {"dense": "#3f4652", "alps": "#d95f0e", "sparsegpt": "#8c4a9e",
         "ours": "#0f8b7e", "alps_sft": "#b08600"}

    def col(lab):
        for k, v in C.items():
            if lab.startswith(k):
                return v
        return "#888"

    for L in args.layers:
        ncol = min(len(prompts), 6)
        fig, axes = plt.subplots(1, ncol, figsize=(3.7 * ncol, 4.2))
        axes = np.atleast_1d(axes)
        for pi in range(ncol):
            ax = axes[pi]
            S = scale[pi].get(L)
            if S is None or len(S) < 4:
                ax.axis("off"); continue
            # basis from the dense scale cloud, so the axes mean "how dense's
            # own generations vary" and displacements are read against that
            pca = PCA(n_components=2, random_state=args.seed).fit(S)
            Z = pca.transform(S)
            ax.scatter(Z[:, 0], Z[:, 1], s=16, alpha=.30, c="#9aa3ad",
                       linewidths=0, label="dense rollouts (scale)")
            for lab in labs:
                B = states[pi][lab][L]
                if len(B) == 0:
                    continue
                Zb = pca.transform(B)
                ax.scatter(Zb[:, 0], Zb[:, 1], s=70, marker="*",
                           c=col(lab), edgecolors="white", linewidths=.8,
                           zorder=5, label=f"CoT via {lab}")
            ax.set_xticks([]); ax.set_yticks([])
            sub = "  ".join(f"{lab.split('_s')[0] if '_s' in lab else lab} "
                            f"{np.mean(results.get(f'L{L}/{lab}', [np.nan])):.2f}"
                            for lab in labs if lab != "dense")
            ax.set_title(f"prompt {pi}\ndisplacement: {sub}", fontsize=8)
        axes[0].legend(loc="upper left", fontsize=7, markerscale=.9, framealpha=.9)
        fig.suptitle(f"The SAME dataset CoT, read by each model (layer {L}). "
                     f"Stars are {args.n_seg} segments of one text; grey is dense's "
                     f"own rollout cloud, drawn as the scale bar.", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        p = os.path.join(args.outdir, f"cot_through_models_L{L}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[cot] wrote {p}", flush=True)

    json.dump(results, open(os.path.join(args.outdir, "cot_displacement.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
