"""Model-level decode with the host cost removed from BOTH sides.

The first end-to-end run put MACKO at 1.17x over dense at 8K tokens, but the
layer study then showed MACKO's wrapper costs ~11us per Linear call -- 2.8ms
per token across 252 of them -- which is most of why its per-token time sat
frozen at 17.0ms while dense drifted from 16.4 to 20.1ms. So that 1.17x was
measured with one side carrying a wrapper the other does not have.

Graphing only MACKO would be the mirror of the same mistake, so both paths are
captured and replayed here. Dense gets faster too; whether the ratio ends up
above or below 1.17x is the question.

A CUDA graph needs every shape and address fixed across replays, so decode runs
against a StaticCache at a pinned cache position rather than HF's default
growing cache. That also means one graph measures decode at ONE sequence
length; the sweep re-captures per length, which is how the KV-cache effect
stays visible.

Usage: bench_macko_graph_e2e.py --dense <dir> --sparse <dir>
"""
import argparse, gc, json, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

import macko_spmv
_MULT = torch.ops.macko_spmv.multiply.default

LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LENGTHS = [512, 2048, 4096, 8192]


class MackoLean(nn.Module):
    """Bias-free Linear on the MACKO kernel, with everything resolvable hoisted
    out of forward: module import, the 5-tuple, and the assert-laden Python
    wrapper are all gone; only the op call remains."""

    def __init__(self, compressed, out_features):
        super().__init__()
        c0, c1, c2, c3, c4 = compressed
        self.register_buffer("c0", c0, persistent=False)
        self.register_buffer("c1", c1, persistent=False)
        self.register_buffer("c2", c2, persistent=False)
        self.c3, self.c4 = c3, c4
        self.out_features = out_features

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] == 1:
            y = _MULT(self.c0, self.c1, self.c2, self.c3, self.c4, flat[0]).unsqueeze(0)
        else:
            y = torch.stack([_MULT(self.c0, self.c1, self.c2, self.c3, self.c4, flat[i])
                             for i in range(flat.shape[0])])
        # Restore the leading dims. Returning [1, out] for a [1, 1, in] input
        # happens to survive an allclose against [1, 1, out] by broadcasting,
        # which is how this went unnoticed in the layer benchmark; inside a
        # model it does not survive.
        return y.reshape(x.shape[:-1] + (self.out_features,))


def compress_model(model):
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or not any(t in name for t in LINEARS):
            continue
        w = mod.weight.data.to(torch.float16)
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1],
                MackoLean(macko_spmv.compress(w), w.shape[0]))
        del w
        n += 1
    gc.collect(); torch.cuda.empty_cache()
    return n


def graph_ms(fn, iters=200):
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


def eager_ms(fn, iters=100, warmup=20):
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


@torch.no_grad()
def measure(model, cfg, length):
    """One decode step against a StaticCache already holding `length` tokens."""
    from transformers import StaticCache
    out = {}
    cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=length + 8,
                        device="cuda", dtype=torch.float16)
    # Fill the cache so attention actually reads `length` keys.
    prompt = torch.randint(0, 10000, (1, length), device="cuda")
    model(prompt, past_key_values=cache,
          cache_position=torch.arange(length, device="cuda"))

    nxt = torch.randint(0, 10000, (1, 1), device="cuda")
    pos = torch.tensor([length], device="cuda")
    step = lambda: model(nxt, past_key_values=cache, cache_position=pos)

    out["eager_ms"] = eager_ms(step)
    try:
        out["graph_ms"] = graph_ms(step)
    except Exception as ex:
        out["graph_ms"] = float("nan")
        out["graph_err"] = f"{type(ex).__name__}: {ex}"[:150]
    del cache, prompt
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print(f"torch {torch.__version__}  gpu = {torch.cuda.get_device_name(0)}")
    res = {}
    for tag, path, sparse in (("dense", a.dense, False), ("macko", a.sparse, True)):
        cfg = AutoConfig.from_pretrained(path)
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
        if sparse:
            t0 = time.perf_counter()
            n = compress_model(m)
            print(f"compressed {n} Linears in {time.perf_counter()-t0:.0f}s")
        print(f"--- {tag} ---")
        for L in LENGTHS:
            r = measure(m, cfg, L)
            res[(tag, L)] = r
            msg = f"  L={L:<6} eager {r['eager_ms']:7.2f}ms  graph {r['graph_ms']:7.2f}ms"
            if "graph_err" in r:
                msg += f"   [{r['graph_err']}]"
            print(msg)
        del m; gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'KV length':<12}{'dense eager':>13}{'macko eager':>13}{'eager':>9}"
          f"{'dense graph':>13}{'macko graph':>13}{'graph':>9}")
    print("-" * 82)
    for L in LENGTHS:
        d, s = res[("dense", L)], res[("macko", L)]
        def r(k):
            x, y = d[k], s[k]
            return f"{x/y:.3f}x" if (y == y and y > 0) else "n/a"
        print(f"{L:<12}{d['eager_ms']:13.2f}{s['eager_ms']:13.2f}{r('eager_ms'):>9}"
              f"{d['graph_ms']:13.2f}{s['graph_ms']:13.2f}{r('graph_ms'):>9}")
    print("\n(>1 means the sparse model on MACKO is faster)")
    print("prior measurements: SpMV kernel 2.50x (A6000), 5-Linear sum under")
    print("graph 1.56x (L40S), full generate() eager 1.17x at 8K tokens (L40S)")

    if a.out:
        json.dump({f"{t}_L{l}": v for (t, l), v in res.items()}, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
