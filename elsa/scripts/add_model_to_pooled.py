#!/usr/bin/env python
"""Add one model to an existing pooled.npz without re-encoding the rest.

policy_divergence_tsne.py re-encodes every model on every run -- about four
hours at 30 prompts -- which is the wrong price for adding a single ablation
checkpoint. This samples the new model, encodes its rollouts with the dense
model (the same shared encoder the rest used), and writes a sidecar npz that
the plotting code merges.

Reuses the prompt cache and writes its own samples cache, so a rerun is free.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from onpolicy_mismatch_diag import generate  # noqa: E402
from policy_divergence_tsne import pool_states  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", required=True,
                    help="existing run dir: uses its prompts.json and pooled_meta.json")
    ap.add_argument("--dense_model", required=True)
    ap.add_argument("--label", required=True, help="e.g. noopd:s50")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    D = args.states_dir
    meta = json.load(open(os.path.join(D, "pooled_meta.json")))
    pc = json.load(open(os.path.join(D, "prompts.json")))
    prompts = pc["prompts"]
    windows = {k: tuple(v) for k, v in meta["windows"].items()}
    layers = meta["layers"]
    # Match the sample count the existing clouds used, or an MMD against them
    # compares differently-sized draws.
    n_s = args.n_samples or meta.get("per_window_samples") or None
    if n_s is None:
        z = np.load(os.path.join(D, "pooled.npz"))
        k = next(k for k in z.files if k.startswith("p0|dense|"))
        n_s = z[k].shape[0]
    print(f"[add] {len(prompts)} prompts x {n_s} samples, windows={windows}, "
          f"layers={layers}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.dense_model, trust_remote_code=True)
    mml = max(len(tok(p, add_special_tokens=False).input_ids)
              for p in prompts) + args.max_new_tokens + 64

    safe = args.label.replace(":", "_")
    cache = os.path.join(D, f"samples_{safe}.json")
    if os.path.exists(cache):
        rolls = json.load(open(cache))
        print(f"[add] rollouts from cache", flush=True)
    else:
        print(f"[add] sampling {args.label} ...", flush=True)
        from vllm import SamplingParams  # noqa: F401  (generate imports its own)
        import policy_divergence_tsne as pdt
        rolls = pdt.sample_many(args.model_path, prompts, n_s,
                                args.max_new_tokens, args.temperature,
                                args.gpu_mem, args.seed, mml)
        json.dump(rolls, open(cache, "w"))
        lens = [len(s) for pr in rolls for s in pr]
        print(f"[add] mean len {np.mean(lens):.0f}", flush=True)

    print("[add] encoding with the dense model ...", flush=True)
    dense = AutoModelForCausalLM.from_pretrained(
        args.dense_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True).to("cuda").eval()
    blob = {}
    for pi, prompt in enumerate(prompts):
        packed, cov = pool_states(dense, tok, prompt, rolls[pi], layers, windows,
                                  "cuda")
        for L in layers:
            for w in windows:
                v = packed[L][w]
                if len(v):
                    blob[f"p{pi}|{args.label}|L{L}|{w}"] = v
        if pi % 5 == 0:
            print(f"[add]   prompt {pi}: {cov}", flush=True)
    del dense
    import gc
    gc.collect(); torch.cuda.empty_cache()

    out = os.path.join(D, f"pooled_extra_{safe}.npz")
    tmp = out + ".tmp.npz"
    np.savez(tmp, **blob)
    os.replace(tmp, out)
    print(f"[add] wrote {out} ({len(blob)} arrays)", flush=True)


if __name__ == "__main__":
    main()
