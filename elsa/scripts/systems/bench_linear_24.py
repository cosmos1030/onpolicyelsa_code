"""Does the 2:4 kernel actually beat the dense kernel on OUR Linear shapes?

The control that makes this measurement mean anything: both sides run the SAME
weight tensor. The dense side holds a [out, in] BF16 tensor that happens to be
half zeros; the sparse side holds that identical tensor compressed by
to_sparse_semi_structured(). Any difference is therefore the kernel alone --
generation length, accuracy and token sequence cannot enter.

M (the token dimension) is swept because decode at batch 1 is M=1, where a
sparse Tensor Core has almost nothing to amortise, while prefill is M in the
hundreds or thousands. A speedup that appears only at large M is still a real
result, but it is a prefill result and must not be reported as a decode one.

If the sparse kernel does not win here, there is no point taking this to a full
model.

Usage: bench_linear_24.py --model <dir> [--iters 50] [--out results.json]
"""
import argparse, glob, json, os
import torch
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured

# One representative weight per distinct shape in a Qwen3 block.
PICK = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.down_proj"]
M_GRID = [1, 8, 32, 128, 512, 2048]


def grab_weights(path, layer=0):
    from safetensors import safe_open
    want = {f"model.layers.{layer}.{p}.weight": p for p in PICK}
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt") as fh:
            for k in fh.keys():
                if k in want:
                    out[want[k]] = fh.get_tensor(k)
        if len(out) == len(PICK):
            break
    return out


def timed(fn, iters, warmup=20):
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
    return s.elapsed_time(e) / iters      # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dev = "cuda"
    print(f"torch {torch.__version__}   gpu = {torch.cuda.get_device_name(0)}")
    try:
        print(f"cuSPARSELt available = {torch.backends.cusparselt.is_available()}")
    except Exception:
        print("cuSPARSELt availability: unknown (falling back to CUTLASS)")

    ws = grab_weights(a.model, a.layer)
    missing = [p for p in PICK if p not in ws]
    if missing:
        raise SystemExit(f"missing weights in {a.model}: {missing}")

    rows = []
    for name in PICK:
        w = ws[name].to(dev).to(torch.bfloat16).contiguous()
        out_f, in_f = w.shape
        zeros = (w == 0).float().mean().item()
        try:
            w_sp = to_sparse_semi_structured(w)
        except Exception as ex:
            print(f"{name} {tuple(w.shape)}: compress FAILED -- {ex}")
            continue

        # Same weight on both paths, so they must agree numerically. Check once
        # before trusting any timing: a kernel that silently computes something
        # else would otherwise look like a speedup.
        x = torch.randn(128, in_f, device=dev, dtype=torch.bfloat16)
        y_d, y_s = F.linear(x, w), F.linear(x, w_sp)
        err = (y_d.float() - y_s.float()).abs().max().item()
        scale = y_d.float().abs().max().item()
        agree = err <= 2e-2 * max(scale, 1.0)

        for M in M_GRID:
            x = torch.randn(M, in_f, device=dev, dtype=torch.bfloat16)
            t_d = timed(lambda: F.linear(x, w), a.iters)
            t_s = timed(lambda: F.linear(x, w_sp), a.iters)
            rows.append(dict(weight=name, out_f=out_f, in_f=in_f, M=M, zeros=zeros,
                             dense_ms=t_d, sparse_ms=t_s, speedup=t_d / t_s,
                             max_abs_err=err, agree=agree))
        del w, w_sp
        torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("nothing measured")

    print(f"\n{'weight':<20}{'shape':>14}{'M':>6}{'dense ms':>11}{'sparse ms':>11}{'speedup':>9}")
    print("-" * 71)
    for r in rows:
        mark = "" if r["agree"] else "  <-- MISMATCH"
        print(f"{r['weight']:<20}{str((r['out_f'], r['in_f'])):>14}{r['M']:>6}"
              f"{r['dense_ms']:11.4f}{r['sparse_ms']:11.4f}{r['speedup']:9.3f}{mark}")

    print("\nspeedup by M (geometric mean over the five shapes):")
    for M in M_GRID:
        sub = [r["speedup"] for r in rows if r["M"] == M]
        if sub:
            g = float(torch.tensor(sub).log().mean().exp())
            print(f"  M = {M:<6} {g:.3f}x")

    bad = sorted({r["weight"] for r in rows if not r["agree"]})
    if bad:
        print(f"\nWARNING: dense and sparse disagree on {bad} -- their timings "
              f"are not comparable.")

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
