"""Depth sweep of the policy-divergence probe (fork of coverage_fixed_vs_rollout.py).

The published probe figure is read at layer 18, which is the middle of
Qwen3-4B's 36 blocks, and that choice was never justified in print. This fork
answers it with data: one forward pass already produces every layer's hidden
states, so encoding at N depths costs the same as encoding at one.

The answer matters beyond bookkeeping. At the FINAL layer every pruned arm
separates from dense at AUC 0.91-0.94 -- SCOUT, w/o rollout refresh, w/o OPD
and ALPS+recovery alike -- and the ordering between them inverts. Last-layer
states sit next to the output distribution, where "this model is pruned" is
trivially decodable, so that depth answers a different question than the one
the ablation asks. Mid-depth states still carry the distinction. Reporting the
whole curve lets a reader pick a depth for their own setup instead of
inheriting ours.

Nothing about the measurement changes but the depth: same rollouts, same
matched-span windows, same prompt-grouped CV.

Usage:
  coverage_layer_sweep.py --states_dir <pool> --models dense,ours:s70,... \
      --layers 4 9 13 18 22 27 31 36 --out sweep.npz
"""
import argparse
import json
import os
import random

import numpy as np
import torch

DENSE = ("/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
         "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
WIN = 256        # tokens per window
N_WIN = 8        # windows per continuation -> 30 prompts x 8 = 240 points a side
SKIP = 256       # drop the opening: every continuation starts "<think> Okay, so"
                 # and those first tokens are near-identical across all sources


@torch.no_grad()
def windows_multi(model, ctx_ids, cont_ids, device, layers, offsets):
    """Pooled states at GIVEN absolute continuation offsets, for EVERY layer in
    `layers`, from a single forward pass.

    Offsets are passed in rather than derived per sequence, because the two
    sides have to be read at the same depth. A window taken at 12.5% of a
    12k-token reference trace sits at token 1500; the same fraction of a
    2k-token rollout sits at token 250, and hidden states move with absolute
    position, so a probe could separate the sources on depth alone.
    """
    need = max(offsets) + WIN
    if len(cont_ids) < need:
        return {L: [] for L in layers}
    ids = ctx_ids + list(cont_ids[:need])
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    hs = model.model(x, output_hidden_states=True).hidden_states
    b = len(ctx_ids)
    return {L: [hs[L][0][b + o: b + o + WIN].float().mean(0).cpu().numpy()
                for o in offsets] for L in layers}


def prompt_bootstrap(fn, groups_a, groups_b, n=2000, seed=0):
    """Resample PROMPTS, not windows. Windows inside one trajectory are highly
    correlated, so treating 240 of them as 240 independent draws overstates
    confidence by roughly sqrt(N_WIN)."""
    rng = np.random.default_rng(seed)
    prompts = np.unique(np.r_[groups_a, groups_b])
    out = []
    for _ in range(n):
        pick = rng.choice(prompts, len(prompts), replace=True)
        v = fn(pick)
        if v is not None:
            out.append(v)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out \
        else (float("nan"), float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", required=True)
    ap.add_argument("--models", default="dense,ours:s70,noopd55:s70,alps_sft:s70")
    ap.add_argument("--n_roll", type=int, default=8,
                    help="rollouts per prompt; windows come from each of them")
    ap.add_argument("--layers", type=int, nargs="+",
                    default=[4, 9, 13, 18, 22, 27, 31, 36],
                    help="hidden_states indices; 0 is the embedding output "
                         "and 36 the final block for Qwen3-4B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dense", default=DENSE)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.dense, trust_remote_code=True)

    D = a.states_dir
    pc = json.load(open(os.path.join(D, "prompts.json")))
    prompts, solutions = pc["prompts"], pc["solutions"]
    ctx = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    # The fixed side: the dataset's own trace for each of THESE prompts.
    sol = [tok(s_, add_special_tokens=False).input_ids for s_ in solutions]
    print(f"[cov] {len(prompts)} prompts | ctx {np.mean([len(c) for c in ctx]):.0f} "
          f"| reference CoT {np.mean([len(s_) for s_ in sol]):.0f} tok", flush=True)

    print("[cov] loading dense encoder ...", flush=True)
    dense = AutoModelForCausalLM.from_pretrained(
        a.dense, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()

    # Per prompt, both sides are cut to the SAME length -- the shorter of the
    # reference trace and the shortest rollout used -- and windows are taken at
    # identical absolute offsets inside that span.
    roll_cache = {}
    for lab in [m for m in a.models.split(",") if m]:
        f = os.path.join(D, f"samples_{lab.replace(':', '_')}.json")
        if os.path.exists(f):
            roll_cache[lab] = json.load(open(f))
    span = []
    rng0 = random.Random(a.seed)
    picks = {}
    for lab, rolls in roll_cache.items():
        picks[lab] = []
        for pi in range(len(prompts)):
            pr = rolls[pi][:]; rng0.shuffle(pr); picks[lab].append(pr[:a.n_roll])
    for pi in range(len(prompts)):
        lens = [len(sol[pi])] + [len(r) for lab in picks for r in picks[lab][pi]]
        span.append(min(lens))
    OFF = {pi: [SKIP + i * WIN for i in range(N_WIN)
                if SKIP + (i + 1) * WIN <= span[pi]] for pi in range(len(prompts))}
    usable = [pi for pi in OFF if OFF[pi]]
    print(f"[cov] matched span per prompt: median {int(np.median(span))} tok, "
          f"{len(usable)}/{len(prompts)} prompts usable, "
          f"{int(np.median([len(OFF[pi]) for pi in usable]))} windows each", flush=True)

    blob, owner = {}, {}
    LAYERS = list(a.layers)

    v = {L: [] for L in LAYERS}; who = []
    for pi in usable:
        w = windows_multi(dense, ctx[pi], sol[pi], dev, LAYERS, OFF[pi])
        for L in LAYERS:
            v[L] += w[L]
        who += [pi] * len(w[LAYERS[0]])
    for L in LAYERS:
        blob[f"L{L}|fixed"] = np.stack(v[L]).astype(np.float16)
    owner["fixed"] = np.array(who)
    print(f"[sweep] fixed: {blob[f'L{LAYERS[0]}|fixed'].shape} per layer "
          f"({len(set(who))} prompts contributed)", flush=True)

    for lab in roll_cache:
        v = {L: [] for L in LAYERS}; who = []
        for pi in usable:
            for r in picks[lab][pi]:
                w = windows_multi(dense, ctx[pi], list(r), dev, LAYERS, OFF[pi])
                for L in LAYERS:
                    v[L] += w[L]
                who += [pi] * len(w[LAYERS[0]])
        for L in LAYERS:
            blob[f"L{L}|{lab}"] = np.stack(v[L]).astype(np.float16)
        owner[lab] = np.array(who)
        print(f"[sweep] {lab}: {blob[f'L{LAYERS[0]}|{lab}'].shape} per layer "
              f"x {len(LAYERS)} layers", flush=True)

    np.savez_compressed(a.out, **blob,
                        **{f"owner_{k}": v_ for k, v_ in owner.items()})
    print(f"[cov] wrote {a.out}", flush=True)

    # Separability is computed by the companion analysis script, which loops
    # over layers -- the original single-layer block does not generalise and is
    # deliberately not carried over.
    print("[sweep] done -- run plot_probe_depth.py on this npz", flush=True)


if __name__ == "__main__":
    main()
