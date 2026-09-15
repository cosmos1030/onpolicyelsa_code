"""How much of MACKO's 2.50x kernel win is the Python wrapper eating?

The end-to-end run pinned MACKO decode at ~17.0ms/token for the whole 8192-token
sequence -- flat to 0.5% while dense drifted 16.4 -> 20.1ms as its KV cache
grew. A number that refuses to move when the work underneath it changes is not
measuring the work; the 252 MackoLayer.forward calls are. This isolates that.

Three layer variants, same compressed weights, same kernel:

  naive     what the end-to-end run used: re-imports the module, rebuilds the
            5-tuple, and calls macko_spmv.multiply (which asserts on each
            buffer) once per forward
  lean      module-level import, tuple built once in __init__, and the op
            invoked directly as torch.ops.macko_spmv.multiply.default
  graph     the lean layer replayed from a captured CUDA graph, which removes
            host-side launch and dispatch entirely

Against a dense nn.Linear measured the same three ways. The gap between 'lean'
and 'graph' is what no amount of Python tidying can reach, and the gap between
'graph' here and the 2.50x microbenchmark is what the surrounding model costs.
"""
import argparse, glob, os, time
import torch
import torch.nn as nn

import macko_spmv
_MULT = torch.ops.macko_spmv.multiply.default

PICK = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.down_proj"]


class MackoNaive(nn.Module):
    def __init__(self, compressed):
        super().__init__()
        c0, c1, c2, c3, c4 = compressed
        self.register_buffer("c0", c0, persistent=False)
        self.register_buffer("c1", c1, persistent=False)
        self.register_buffer("c2", c2, persistent=False)
        self.c3, self.c4 = c3, c4

    def forward(self, x):
        import macko_spmv
        cm = (self.c0, self.c1, self.c2, self.c3, self.c4)
        flat = x.reshape(-1, x.shape[-1])
        y = macko_spmv.multiply(cm, flat[0]).unsqueeze(0)
        return y.reshape(x.shape[:-1] + (y.shape[-1],))


class MackoLean(nn.Module):
    def __init__(self, compressed):
        super().__init__()
        c0, c1, c2, c3, c4 = compressed
        self.register_buffer("c0", c0, persistent=False)
        self.register_buffer("c1", c1, persistent=False)
        self.register_buffer("c2", c2, persistent=False)
        self.c3, self.c4 = c3, c4

    def forward(self, x):
        # x is [1, 1, in] or [1, in] at batch-1 decode; the kernel wants a 1-D
        # vector, and everything else here is already resolved at init.
        return _MULT(self.c0, self.c1, self.c2, self.c3, self.c4,
                     x.reshape(-1)).unsqueeze(0)


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


def eager_ms(fn, iters=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def graph_ms(fn, iters=300):
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(10):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(30):
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

    print(f"torch {torch.__version__}  gpu = {torch.cuda.get_device_name(0)}")
    ws = grab(a.model, a.layer, PICK)

    print(f"\n{'weight':<18}{'shape':>14}"
          f"{'dense eager':>12}{'dense graph':>12}"
          f"{'naive':>9}{'lean':>9}{'graph':>9}"
          f"{'lean/dns':>10}{'graph/dns':>11}")
    print("-" * 104)

    tot = {k: 0.0 for k in ("de", "dg", "nv", "ln", "gr")}
    for n in PICK:
        w = ws[n].cuda().to(torch.float16).contiguous()
        x = torch.randn(1, 1, w.shape[1], device="cuda", dtype=torch.float16)
        lin = nn.Linear(w.shape[1], w.shape[0], bias=False).cuda().half()
        lin.weight.data.copy_(w)
        comp = macko_spmv.compress(w)
        nv, ln = MackoNaive(comp).cuda(), MackoLean(comp).cuda()

        # Correctness before timing: all three must agree with the dense Linear.
        ref = lin(x).float()
        for tag, mod in (("naive", nv), ("lean", ln)):
            err = (ref - mod(x).float()).abs().max().item()
            if err > 2e-2 * max(ref.abs().max().item(), 1.0):
                print(f"  {n} {tag}: MISMATCH max_err={err:.3e}")

        de = eager_ms(lambda: lin(x))
        dg = graph_ms(lambda: lin(x))
        nvm = eager_ms(lambda: nv(x))
        lnm = eager_ms(lambda: ln(x))
        grm = graph_ms(lambda: ln(x))
        for k, v in zip(("de", "dg", "nv", "ln", "gr"), (de, dg, nvm, lnm, grm)):
            tot[k] += v
        print(f"{n:<18}{str(tuple(w.shape)):>14}{de:12.4f}{dg:12.4f}"
              f"{nvm:9.4f}{lnm:9.4f}{grm:9.4f}{de/lnm:9.3f}x{dg/grm:10.3f}x")
        del w, lin, comp, nv, ln
        torch.cuda.empty_cache()

    print(f"\nsum over the 5 shapes (ms): dense eager {tot['de']:.4f}  "
          f"dense graph {tot['dg']:.4f}  naive {tot['nv']:.4f}  "
          f"lean {tot['ln']:.4f}  graph {tot['gr']:.4f}")
    print(f"  lean vs dense eager : {tot['de']/tot['ln']:.3f}x")
    print(f"  graph vs dense graph: {tot['dg']/tot['gr']:.3f}x")
    print(f"  wrapper cost removed by graphing: "
          f"{(tot['ln'] - tot['gr']) / 5 * 1000:.0f} us per Linear call")
    print(f"\nQwen3-4B has 252 prunable Linears, so a per-call wrapper cost of X us")
    print(f"is 252*X us on every decode token.")


if __name__ == "__main__":
    main()
