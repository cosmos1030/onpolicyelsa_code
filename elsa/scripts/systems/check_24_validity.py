"""Is a checkpoint's 2:4 mask the one the Sparse Tensor Cores actually want?

A 2:4 kernel reads four CONTIGUOUS elements along the REDUCTION dimension and
requires exactly two of them to be zero. For nn.Linear the weight is stored
[out_features, in_features] and the reduction dimension is in_features, i.e.
the last axis. A mask that is 50% sparse "in groups of 4" along any other axis
-- or an unstructured 50% mask -- is not loadable by the kernel at all, so this
has to be checked before any timing work.

torch's BF16/FP16 semi-structured path additionally wants both matrix
dimensions to be multiples of 64.

Usage: check_24_validity.py <model_dir_or_repo> [--limit N]
"""
import argparse, json, os, sys
import torch

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_weights(path, limit=None):
    from safetensors import safe_open
    import glob
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise SystemExit(f"no safetensors under {path}")
    n = 0
    for f in files:
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                if not k.endswith(".weight"):
                    continue
                if not any(t in k for t in TARGETS):
                    continue
                yield k, fh.get_tensor(k)
                n += 1
                if limit and n >= limit:
                    return


def frac_groups_of_two(w, dim):
    """Fraction of contiguous 4-groups along `dim` that hold exactly 2 nonzeros."""
    w = w.movedim(dim, -1)
    if w.shape[-1] % 4:
        return float("nan")
    nz = (w != 0).reshape(-1, w.shape[-1] // 4, 4).sum(-1)
    return (nz == 2).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args()

    rows, bad64 = [], []
    for name, w in load_weights(a.path, a.limit):
        w = w.float()
        out_f, in_f = w.shape
        rows.append((
            name, tuple(w.shape),
            (w == 0).float().mean().item(),
            frac_groups_of_two(w, 1),   # along in_features  <- the one that matters
            frac_groups_of_two(w, 0),   # along out_features <- reported to diagnose a transposed mask
        ))
        if out_f % 64 or in_f % 64:
            bad64.append((name, (out_f, in_f)))

    if not rows:
        raise SystemExit("no prunable Linear weights found")

    print(f"{'weight':<46}{'shape':>18}{'zeros':>8}{'2:4 in_f':>10}{'2:4 out_f':>11}")
    print("-" * 93)
    for name, shape, z, a_in, a_out in rows:
        print(f"{name:<46}{str(shape):>18}{z:8.3f}{a_in:10.3f}{a_out:11.3f}")

    ok_in = all(r[3] > 0.999 for r in rows)
    ok_out = all(r[4] > 0.999 for r in rows)
    print()
    if ok_in:
        print("VALID: every 4-group along in_features has exactly 2 nonzeros "
              "-- this is what to_sparse_semi_structured() expects.")
    elif ok_out:
        print("TRANSPOSED: the 2:4 structure runs along out_features, not the "
              "reduction dim. The kernel cannot use this mask as stored.")
    else:
        print("NOT 2:4: the mask is 50% sparse but not in hardware 2:4 groups "
              "on either axis. Semi-structured execution is not possible.")
    if bad64:
        print(f"\nDIM WARNING: {len(bad64)} weight(s) not a multiple of 64 on both axes:")
        for n, s in bad64[:5]:
            print(f"  {n} {s}")
    sys.exit(0 if ok_in else 1)


if __name__ == "__main__":
    main()
