"""Fold the SCOUT-w/o-OPD clouds into the figure's self_gen npz.

The figure reads one flat file, pooled_L18_late_fp16.npz, holding 6 prompts x 96
rollouts at full 2560 dims. add_model_to_pooled.py does not touch that file -- it
writes a sidecar, pooled_extra_<label>.npz, next to the source pool, carrying
every layer and window. So this copies the existing arrays through byte for byte
and appends only the L18/late slices of the new label.

Copying through rather than rebuilding from n6_k96_base/pooled.npz is the point:
the published curves must not move because a fourth method was added. Any key
already present wins, and the script refuses to overwrite one.

Usage: merge_noopd55.py [--labels noopd55:s50,noopd55:s60,noopd55:s70]
"""
import argparse
import os
import shutil

import numpy as np

POOL = "/home1/doyoonkim/projects/elsa/logs/policy_divergence/n6_k96_base"
TARGET = "plotdata/plotdata/self_gen/pooled_L18_late_fp16.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",
                    default="noopd55:s50,noopd55:s60,noopd55:s70")
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--target", default=TARGET)
    args = ap.parse_args()
    labels = [x for x in args.labels.split(",") if x]

    z = np.load(args.target)
    keep = {k: z[k] for k in z.files}
    before = len(keep)
    n_prompts = len({k.split("|")[0] for k in keep})

    added = 0
    for lab in labels:
        f = os.path.join(args.pool, f"pooled_extra_{lab.replace(':', '_')}.npz")
        if not os.path.exists(f):
            raise SystemExit(f"missing sidecar: {f}")
        e = np.load(f)
        for pi in range(n_prompts):
            k = f"p{pi}|{lab}|L18|late"
            if k not in e.files:
                raise SystemExit(f"{os.path.basename(f)} has no {k}")
            if k in keep:
                raise SystemExit(f"{k} already in target -- refusing to overwrite")
            v = e[k]
            ref = keep[f"p{pi}|dense|L18|late"]
            if v.shape[1] != ref.shape[1]:
                raise SystemExit(f"{k} is {v.shape}, dense is {ref.shape}")
            if v.shape[0] != ref.shape[0]:
                # An MMD between differently sized draws is not the same
                # quantity, so this is a stop, not a warning.
                raise SystemExit(f"{k} has {v.shape[0]} rollouts, dense has "
                                 f"{ref.shape[0]}")
            keep[k] = v.astype(np.float16)
            added += 1
        print(f"  + {lab}: {n_prompts} prompts")

    shutil.copy2(args.target, args.target + ".bak")
    tmp = args.target + ".tmp.npz"
    np.savez(tmp, **keep)
    os.replace(tmp, args.target)
    print(f"{before} -> {len(keep)} arrays (+{added}); "
          f"backup at {os.path.basename(args.target)}.bak")

    z2 = np.load(args.target)
    models = sorted({k.split("|")[1] for k in z2.files})
    print("models:", models)


if __name__ == "__main__":
    main()
