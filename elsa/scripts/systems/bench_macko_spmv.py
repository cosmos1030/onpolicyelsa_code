"""MACKO-SpMV vs cuBLAS on Qwen3 weight shapes at high unstructured sparsity.

Same shape of question as the 2:4 work, and the same control: one weight
tensor, two ways of multiplying by it. The dense side is a [out, in] fp16
tensor that happens to be mostly zeros; the sparse side is that identical
tensor in MACKO's compressed format.

The regime is deliberately the one 2:4 lost in. MACKO is SpMV -- matrix times
VECTOR -- so it only addresses M=1, which is batch-1 decode. At M=1 the 2:4
kernel measured 0.58x, i.e. slower than dense. If MACKO wins here it covers
exactly the gap 2:4 cannot.

Two things to watch:
  * The library is tuned on consumer cards (2080/3090/4090) and its README
    says server GPUs need separate optimisation. Run this where it is tuned
    before concluding anything about the format.
  * Reported gains scale with sparsity: ~1.3-1.5x at 50% density, 3.5-4.5x at
    10%. 80% sparsity is 20% density, so the interesting range.

Usage: bench_macko_spmv.py --model <dir> [--layer 0]
"""
import argparse, glob, os, time
import torch

PICK = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.down_proj"]


def grab(path, layer, names):
    from safetensors import safe_open
    want = {f"model.layers.{layer}.{n}.weight": n for n in names}
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                if k in want:
                    out[want[k]] = fh.get_tensor(k)
        if len(out) == len(names):
            break
    return out


def timed(fn, iters=200, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()

    import macko_spmv     # JIT-compiles the CUDA kernels on first import
    print(f"torch {torch.__version__}  gpu = {torch.cuda.get_device_name(0)}")
    print("macko_spmv imported (kernels built)")

    ws = grab(a.model, a.layer, PICK)
    missing = [p for p in PICK if p not in ws]
    if missing:
        raise SystemExit(f"missing weights: {missing}")

    print(f"\n{'weight':<20}{'shape':>15}{'density':>9}"
          f"{'dense ms':>11}{'macko ms':>11}{'speedup':>9}{'max err':>11}")
    print("-" * 86)

    rows = []
    for n in PICK:
        w = ws[n].cuda().to(torch.float16).contiguous()
        out_f, in_f = w.shape
        density = (w != 0).float().mean().item()
        x = torch.randn(in_f, device="cuda", dtype=torch.float16)

        try:
            cm = macko_spmv.compress(w)
        except Exception as ex:
            print(f"{n:<20}{str((out_f, in_f)):>15}  compress FAILED -- "
                  f"{type(ex).__name__}: {ex}")
            continue

        # Same weight both ways, so the outputs must match before any timing is
        # worth reading.
        y_d = torch.mv(w, x)
        y_s = macko_spmv.multiply(cm, x)
        err = (y_d.float() - y_s.float()).abs().max().item()
        scale = max(y_d.float().abs().max().item(), 1.0)
        ok = err <= 2e-2 * scale

        t_d = timed(lambda: torch.mv(w, x), a.iters)
        t_s = timed(lambda: macko_spmv.multiply(cm, x), a.iters)
        rows.append((n, out_f, in_f, density, t_d, t_s, t_d / t_s, err, ok))
        mark = "" if ok else "  <-- MISMATCH"
        print(f"{n:<20}{str((out_f, in_f)):>15}{density:9.3f}"
              f"{t_d:11.4f}{t_s:11.4f}{t_d/t_s:9.3f}{err:11.2e}{mark}")
        del w, cm
        torch.cuda.empty_cache()

    if rows:
        g = float(torch.tensor([r[6] for r in rows]).log().mean().exp())
        print(f"\ngeometric mean speedup over {len(rows)} shapes: {g:.3f}x")
        bad = [r[0] for r in rows if not r[8]]
        if bad:
            print(f"WARNING: outputs disagree on {bad}; those timings are not "
                  f"comparable.")


if __name__ == "__main__":
    main()
