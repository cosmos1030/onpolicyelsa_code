"""Why is the 2:4 path a flat ~0.7ms regardless of problem size?

The first microbenchmark measured sparse time as constant across a 600x range
of FLOPs (k_proj at M=1 vs gate_proj at M=2048). A GEMM cannot do that, so the
number was not a GEMM: something fixed per call dominated it. This separates
the candidates.

Three ways of issuing the same 2:4 matmul:
  subclass   F.linear(x, w_sp)          -- goes through SparseSemiStructuredTensor's
                                            Python __torch_dispatch__ (prototype)
  raw op     torch._cslt_sparse_mm(...)  -- the cuSPARSELt op the subclass ends up
                                            calling, with no Python wrapper
  cudagraph  the subclass call, replayed from a captured CUDA graph, which
             removes host-side launch and dispatch cost entirely

If raw-op and cudagraph scale with M while the subclass stays flat, the flat
number was dispatch overhead and the kernel itself was never measured.
"""
import argparse, glob, os
import torch
import torch.nn.functional as F
from torch.sparse import to_sparse_semi_structured

M_GRID = [1, 32, 512, 2048]


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


def timed(fn, iters=100, warmup=30):
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


def timed_graph(fn, iters=100):
    """Capture fn into a CUDA graph and time replays, so host cost is excluded."""
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(10):
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
    a = ap.parse_args()

    names = ["self_attn.k_proj", "mlp.gate_proj"]   # smallest and largest
    ws = grab(a.model, a.layer, names)
    print(f"torch {torch.__version__}  gpu={torch.cuda.get_device_name(0)}  "
          f"cuSPARSELt={torch.backends.cusparselt.is_available()}")
    print(f"\n{'weight':<18}{'M':>6}{'dense':>10}{'subclass':>11}{'raw op':>10}"
          f"{'cudagraph':>12}{'raw vs dense':>14}")
    print("-" * 81)

    for n in names:
        w = ws[n].cuda().to(torch.bfloat16).contiguous()
        w_sp = to_sparse_semi_structured(w)
        # The compressed buffer the cuSPARSELt op wants, pulled off the subclass.
        packed = getattr(w_sp, "packed", None)
        for M in M_GRID:
            x = torch.randn(M, w.shape[1], device="cuda", dtype=torch.bfloat16)
            t_dense = timed(lambda: F.linear(x, w))
            t_sub = timed(lambda: F.linear(x, w_sp))
            t_raw = float("nan")
            if packed is not None and hasattr(torch, "_cslt_sparse_mm"):
                try:
                    xt = x.t().contiguous()
                    timed(lambda: torch._cslt_sparse_mm(packed, xt), iters=5, warmup=2)
                    t_raw = timed(lambda: torch._cslt_sparse_mm(packed, xt))
                except Exception as ex:
                    print(f"   raw op unavailable for {n} M={M}: {type(ex).__name__}: {ex}")
            try:
                t_g = timed_graph(lambda: F.linear(x, w_sp))
            except Exception as ex:
                t_g = float("nan")
                print(f"   cudagraph failed for {n} M={M}: {type(ex).__name__}")
            ratio = t_dense / t_raw if t_raw == t_raw and t_raw > 0 else float("nan")
            print(f"{n:<18}{M:>6}{t_dense:10.4f}{t_sub:11.4f}{t_raw:10.4f}"
                  f"{t_g:12.4f}{ratio:14.3f}")
        del w, w_sp
        torch.cuda.empty_cache()

    print("\nRead: if 'subclass' is flat while 'raw op' and 'cudagraph' grow with M,")
    print("the flat column was Python dispatch cost, not the 2:4 kernel.")


if __name__ == "__main__":
    main()
