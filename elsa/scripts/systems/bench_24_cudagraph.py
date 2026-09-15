"""2:4 vs dense with the host cost removed from BOTH sides.

The first pass measured the sparse path at a flat ~0.7ms for every shape and
every M, which is impossible for a GEMM; the follow-up showed why -- that was
SparseSemiStructuredTensor's prototype Python __torch_dispatch__, not the
kernel. Under a captured CUDA graph the sparse path scales with M and comes in
under dense at large M.

But that comparison was rigged: only the sparse side was graphed. Dense pays a
launch cost too -- gate_proj measures the same 0.038ms at M=1 and M=32, i.e. it
is sitting on its own floor -- so exempting only one side inflates the winner.
Here both sides are captured and replayed identically, and the eager numbers
are kept alongside so the size of the host cost stays visible.

Usage: bench_24_cudagraph.py --model <dir> [--out results.json]
"""
import argparse, glob, json, os
import torch
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured

PICK = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.down_proj"]
M_GRID = [1, 8, 32, 128, 512, 2048, 4096]


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


def eager_ms(fn, iters=100, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def graph_ms(fn, iters=200):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ws = grab(a.model, a.layer, PICK)
    missing = [p for p in PICK if p not in ws]
    if missing:
        raise SystemExit(f"missing: {missing}")

    print(f"torch {torch.__version__}  gpu={torch.cuda.get_device_name(0)}  "
          f"cuSPARSELt={torch.backends.cusparselt.is_available()}")
    print(f"\n{'weight':<18}{'shape':>14}{'M':>6}"
          f"{'dense g':>10}{'2:4 g':>10}{'speedup':>9}"
          f"{'dense eager':>13}{'2:4 eager':>11}")
    print("-" * 91)

    rows = []
    for n in PICK:
        w = ws[n].cuda().to(torch.bfloat16).contiguous()
        w_sp = to_sparse_semi_structured(w)
        x0 = torch.randn(128, w.shape[1], device="cuda", dtype=torch.bfloat16)
        err = (F.linear(x0, w).float() - F.linear(x0, w_sp).float()).abs().max().item()
        scale = F.linear(x0, w).float().abs().max().item()
        agree = err <= 2e-2 * max(scale, 1.0)

        for M in M_GRID:
            x = torch.randn(M, w.shape[1], device="cuda", dtype=torch.bfloat16)
            try:
                gd = graph_ms(lambda: F.linear(x, w))
                gs = graph_ms(lambda: F.linear(x, w_sp))
            except Exception as ex:
                print(f"{n} M={M}: graph capture failed -- {type(ex).__name__}: {ex}")
                continue
            ed = eager_ms(lambda: F.linear(x, w))
            es = eager_ms(lambda: F.linear(x, w_sp))
            rows.append(dict(weight=n, out_f=w.shape[0], in_f=w.shape[1], M=M,
                             dense_graph_ms=gd, sparse_graph_ms=gs, speedup=gd / gs,
                             dense_eager_ms=ed, sparse_eager_ms=es, agree=agree))
            mark = "" if agree else "  <-- MISMATCH"
            print(f"{n:<18}{str(tuple(w.shape)):>14}{M:>6}"
                  f"{gd:10.4f}{gs:10.4f}{gd/gs:9.3f}{ed:13.4f}{es:11.4f}{mark}")
        del w, w_sp
        torch.cuda.empty_cache()

    print("\n2:4 speedup vs dense, both under CUDA graph (geometric mean over shapes):")
    for M in M_GRID:
        sub = [r["speedup"] for r in rows if r["M"] == M]
        if sub:
            g = float(torch.tensor(sub).log().mean().exp())
            print(f"  M = {M:<6} {g:.3f}x")

    print("\nhost cost hidden by the graph (eager - graph, ms):")
    for M in (1, 2048):
        for side in ("dense", "sparse"):
            sub = [r[f"{side}_eager_ms"] - r[f"{side}_graph_ms"] for r in rows if r["M"] == M]
            if sub:
                print(f"  M={M:<6} {side:<7} mean {sum(sub)/len(sub):7.4f}")

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
