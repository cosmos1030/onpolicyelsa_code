"""End-to-end 2:4 vs dense on the SAME Qwen3-4B checkpoint.

Same control as the Linear benchmark: one set of weights, two execution paths.
The dense path keeps the [out, in] BF16 tensors that happen to be half zeros;
the sparse path replaces every prunable Linear's weight with its
to_sparse_semi_structured() compression. Identical model, identical tokens, so
nothing about generation length or accuracy can enter the number.

Measured separately, because they behave nothing alike:

  prefill   one forward over a full prompt. The token dimension is
            batch x seqlen, i.e. hundreds to thousands -- the regime where the
            Linear benchmark showed 1.1-1.2x.
  decode    one forward of a single new token against a filled KV cache. The
            token dimension is just the batch size, and at batch 1 the Linear
            benchmark showed 0.58x, i.e. 2:4 LOSES.

Both are timed eager and under a captured CUDA graph. That distinction is not
academic here: the sparse path pays ~0.7ms of Python dispatch per Linear call,
and this model has 7 prunable Linears in each of 36 blocks, so eager decode
carries roughly a quarter second per token of pure overhead. Reporting only the
graphed number would overstate what a normal HF generate() call delivers;
reporting only the eager number would blame the kernel for a wrapper.

Usage: bench_model_24.py --model <dir> [--out results.json]
"""
import argparse, gc, json, time
import torch
import torch.nn as nn
from torch.sparse import to_sparse_semi_structured
from transformers import AutoConfig, AutoModelForCausalLM

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def sparsify_(model):
    """Swap every prunable Linear weight for its 2:4 compressed form, in place."""
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(t in name for t in TARGETS):
            w = mod.weight.data
            if w.shape[0] % 64 or w.shape[1] % 64:
                continue
            mod.weight = nn.Parameter(to_sparse_semi_structured(w.contiguous()),
                                      requires_grad=False)
            n += 1
    return n


def eager_ms(fn, iters, warmup=20):
    """20 warmup iterations, not 3.

    The first pass used 3, which is thin enough that cuBLAS algorithm
    selection, allocator growth and CUDA context setup can still be inside the
    measured window. That will not explain a 5.5x gap (sparse eager 182ms vs
    dense 33ms), but it is exactly the kind of thing that can move a borderline
    number like the 0.94x prefill ratio, so it is not worth leaving thin."""
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
    # Median, not mean: one stray slow iteration (a page fault, another tenant
    # on the card) should not decide the comparison.
    return ts[len(ts) // 2]


def graph_ms(fn, iters=100):
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


def build(path, sparse):
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").cuda().eval()
    n = sparsify_(m) if sparse else 0
    return m, n


@torch.no_grad()
def measure(model, batch, seqlen, vocab, decode_iters=20):
    out = {}
    ids = torch.randint(0, vocab, (batch, seqlen), device="cuda")
    # -- prefill --
    out["prefill_eager_ms"] = eager_ms(lambda: model(ids), iters=20)
    try:
        out["prefill_graph_ms"] = graph_ms(lambda: model(ids), iters=50)
    except Exception as ex:
        out["prefill_graph_ms"] = float("nan")
        out["prefill_graph_err"] = f"{type(ex).__name__}: {ex}"[:120]
    # -- decode: one new token against a cache already holding `seqlen` --
    pref = model(ids, use_cache=True)
    cache = pref.past_key_values
    nxt = torch.randint(0, vocab, (batch, 1), device="cuda")
    step = lambda: model(nxt, past_key_values=cache, use_cache=True)
    try:
        out["decode_eager_ms"] = eager_ms(step, iters=decode_iters)
    except Exception as ex:
        out["decode_eager_ms"] = float("nan")
        out["decode_eager_err"] = f"{type(ex).__name__}: {ex}"[:140]
    try:
        out["decode_graph_ms"] = graph_ms(step, iters=100)
    except Exception as ex:
        out["decode_graph_ms"] = float("nan")
        out["decode_graph_err"] = f"{type(ex).__name__}: {ex}"[:120]
    del pref, cache
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(a.model)
    vocab = cfg.vocab_size
    print(f"torch {torch.__version__}  gpu={torch.cuda.get_device_name(0)}  "
          f"cuSPARSELt={torch.backends.cusparselt.is_available()}")
    print(f"model: {a.model}  layers={cfg.num_hidden_layers}  vocab={vocab}")

    grid = [(1, 512), (1, 2048), (8, 512)]
    results = {}
    for mode in ("dense", "sparse"):
        model, nlin = build(a.model, mode == "sparse")
        torch.cuda.reset_peak_memory_stats()
        print(f"\n--- {mode} execution"
              + (f" ({nlin} Linears compressed)" if mode == "sparse" else "") + " ---")
        for batch, seqlen in grid:
            r = measure(model, batch, seqlen, vocab)
            r["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            results[(mode, batch, seqlen)] = r
            print(f"  b={batch} L={seqlen:<5} prefill eager {r['prefill_eager_ms']:8.2f}ms "
                  f"graph {r['prefill_graph_ms']:8.2f}ms | decode eager "
                  f"{r['decode_eager_ms']:7.2f}ms graph {r['decode_graph_ms']:7.2f}ms")
            for k in ("prefill_graph_err", "decode_eager_err", "decode_graph_err"):
                if k in r:
                    print(f"      {k}: {r[k]}")
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'config':<16}{'prefill eager':>15}{'prefill graph':>15}"
          f"{'decode eager':>14}{'decode graph':>14}")
    print("-" * 74)
    for batch, seqlen in grid:
        d, s = results[("dense", batch, seqlen)], results[("sparse", batch, seqlen)]
        def sp(key):
            x, y = d[key], s[key]
            return f"{x/y:.3f}x" if y == y and y > 0 else "n/a"
        print(f"b={batch} L={seqlen:<9}{sp('prefill_eager_ms'):>15}{sp('prefill_graph_ms'):>15}"
              f"{sp('decode_eager_ms'):>14}{sp('decode_graph_ms'):>14}")
    print("\n(>1 means 2:4 execution is faster than dense execution of the same weights)")

    if a.out:
        json.dump({f"{m}_b{b}_L{l}": v for (m, b, l), v in results.items()},
                  open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
