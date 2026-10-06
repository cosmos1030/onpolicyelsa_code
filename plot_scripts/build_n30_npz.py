"""Build the 30-prompt companion to pooled_L18_late_fp16.npz.

The figure's default file holds 6 prompts x 96 rollouts. Six prompts is enough
to show ALPS and SparseGPT pulling away by half an order of magnitude, but not
to separate SCOUT from SCOUT-w/o-OPD, whose error bars overlap there. The
n30_k64_core pool answers the same question with 30 prompts x 64 rollouts --
five times the prompts, so the SEM over prompts shrinks by about sqrt(5).

Same layout as the file it sits beside: L18/late only, full 2560 dims, float16,
keys "p{i}|{model}|L18|late". Models added after the pool was built live in
sidecar pooled_extra_*.npz files, and noopd55 is one of them.

Fewer rollouts per prompt (64 vs 96) is not a problem as long as every model in
a comparison draws the same number, which the pool guarantees -- an MMD between
differently sized draws is not the same quantity. It does make each individual
cloud slightly noisier; the gain is in the prompt-level average.

Usage: build_n30_npz.py [--out plotdata/plotdata/self_gen/pooled_L18_late_fp16_n30.npz]
"""
import argparse
import glob
import os

import numpy as np

POOL = "/home1/doyoonkim/projects/elsa/logs/policy_divergence/n30_k64_core"
# alps_sft is in the pool but not in any figure; teacher is a single-row
# landmark the MMD cannot use. Both are carried through anyway so the file
# stands on its own.
WANT = (["dense", "teacher"]
        + [f"{f}:{s}" for f in ("alps", "sparsegpt", "alps_sft", "ours", "noopd55")
           for s in ("s50", "s60", "s70")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--out",
                    default="plotdata/plotdata/self_gen/pooled_L18_late_fp16_n30.npz")
    args = ap.parse_args()

    z = np.load(os.path.join(args.pool, "pooled.npz"))
    sidecars = [np.load(f) for f in
                sorted(glob.glob(os.path.join(args.pool, "pooled_extra_*.npz")))]

    def fetch(k):
        if k in z.files:
            return z[k]
        for e in sidecars:
            if k in e.files:
                return e[k]
        return None

    n_prompts = len({k.split("|")[0] for k in z.files})
    keep, found = {}, {}
    for pi in range(n_prompts):
        for lab in WANT:
            v = fetch(f"p{pi}|{lab}|L18|late")
            if v is None:
                continue
            keep[f"p{pi}|{lab}|L18|late"] = v.astype(np.float16)
            found[lab] = found.get(lab, 0) + 1

    print(f"prompts: {n_prompts}")
    for lab in WANT:
        n = found.get(lab, 0)
        mark = "  " if n == n_prompts else "!!"
        print(f" {mark} {lab:<18} {n}/{n_prompts} prompts")

    # Rollouts that never reach the late window (continuation tokens 1024-2048)
    # have no row to pool, so a model that emits short answers loses samples.
    # That is a property of the degraded model, not a defect: the published
    # 6-prompt file has the same shortfall on alps:s70 (71 and 77 of 96 on the
    # first two prompts). Report it rather than repair it -- dropping the
    # affected prompts would silently favour the models that collapse.
    full_n = max(keep[k].shape[0] for k in keep if "|teacher|" not in k)
    short = {}
    for lab in found:
        if lab == "teacher":
            continue
        ns = [keep[f"p{pi}|{lab}|L18|late"].shape[0] for pi in range(n_prompts)
              if f"p{pi}|{lab}|L18|late" in keep]
        if any(n < full_n for n in ns):
            short[lab] = [n for n in ns if n < full_n]
    print(f"rollouts per prompt: {full_n}")
    for lab, ns in short.items():
        print(f"  short: {lab} -> {ns} (of {full_n}) on "
              f"{len(ns)}/{n_prompts} prompts")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp.npz"
    np.savez(tmp, **keep)
    os.replace(tmp, args.out)
    print(f"wrote {args.out} ({len(keep)} arrays, "
          f"{os.path.getsize(args.out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
