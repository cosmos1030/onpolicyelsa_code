"""End-to-end batch-1 decode: dense Qwen3-4B vs an 80%-sparse model on MACKO.

This is the deployment comparison, so unlike the kernel microbenchmark the two
sides are NOT the same weights -- one is the original dense model, the other is
the pruned one. That is the point: the question is what a user gains by
deploying the sparse model, and the kernel gain has to survive everything the
model does that is not a Linear.

Decode only, and at batch 1. MACKO is SpMV -- matrix times vector -- so its
CustomLayer multiplies one row at a time; at batch 1 decode that is exactly one
row per Linear per token, which is the regime it is built for. Prefill would
run that Python loop once per prompt token and is not what this measures, so
the prompt is kept short and its cost is reported separately rather than
folded in.

Generation length is FIXED (EOS ignored). A pruned model that rambles would
otherwise show up as a latency regression here, which is a real deployment
concern but a different measurement -- it belongs with the accuracy numbers,
not with a kernel comparison.

Per-token latency is recorded for EVERY step rather than averaged over a short
run, because the speedup is not expected to be constant in the generation
length and our real generations are long (MATH-500 averages ~7,970 output
tokens, and at 70% sparsity every rollout hits the 8192 cap). Two effects pull
opposite ways as the sequence grows: attention reads a KV cache that grows
linearly and MACKO does not touch it, which dilutes the gain; while the
per-token Python dispatch cost of MackoLayer is fixed, so it shrinks as a
share of a heavier step. Which wins is an empirical question, so the run
reports the ratio in buckets across the whole sequence instead of a single
number.

Usage: bench_macko_e2e.py --dense <dir> --sparse <dir> [--new_tokens 128]
"""
import argparse, gc, json, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class MackoLayer(nn.Module):
    """Drop-in for a bias-free nn.Linear backed by the MACKO SpMV kernel.

    The row loop is inherent: the kernel is matrix-VECTOR, so a [T, in] input
    costs T kernel launches. At batch-1 decode T is 1, which is the case this
    benchmark is about.
    """

    def __init__(self, compressed, out_features):
        super().__init__()
        c0, c1, c2, c3, c4 = compressed
        self.register_buffer("c0", c0, persistent=False)
        self.register_buffer("c1", c1, persistent=False)
        self.register_buffer("c2", c2, persistent=False)
        self.c3, self.c4 = c3, c4
        self.out_features = out_features

    def forward(self, x):
        import macko_spmv
        cm = (self.c0, self.c1, self.c2, self.c3, self.c4)
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] == 1:
            y = macko_spmv.multiply(cm, flat[0]).unsqueeze(0)
        else:
            y = torch.stack([macko_spmv.multiply(cm, flat[i])
                             for i in range(flat.shape[0])])
        return y.reshape(x.shape[:-1] + (y.shape[-1],))


def compress_model(model):
    import macko_spmv
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or not any(t in name for t in LINEARS):
            continue
        w = mod.weight.data.to(torch.float16)
        comp = macko_spmv.compress(w)
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], MackoLayer(comp, w.shape[0]))
        del w
        n += 1
    gc.collect(); torch.cuda.empty_cache()
    return n


@torch.no_grad()
def run(model, ids, new_tokens, warmup_tokens=8):
    """Prefill once, then time a fixed number of single-token decode steps."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1e3

    cache = out.past_key_values
    nxt = out.logits[:, -1:].argmax(-1)
    for _ in range(warmup_tokens):          # let clocks and allocator settle
        o = model(nxt, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        nxt = o.logits[:, -1:].argmax(-1)

    torch.cuda.synchronize()
    per_token = []
    for _ in range(new_tokens):
        t0 = time.perf_counter()
        o = model(nxt, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        per_token.append((time.perf_counter() - t0) * 1e3)
        cache = o.past_key_values
        nxt = o.logits[:, -1:].argmax(-1)    # EOS ignored on purpose

    srt = sorted(per_token)
    med = srt[len(srt) // 2]
    return dict(prefill_ms=prefill_ms, decode_ms_median=med,
                decode_tok_s=1000.0 / med,
                decode_ms_p10=srt[len(srt) // 10],
                decode_ms_p90=srt[(9 * len(srt)) // 10],
                per_token_ms=per_token,          # in generation order
                peak_gb=torch.cuda.max_memory_allocated() / 1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--new_tokens", type=int, default=8192)
    ap.add_argument("--prompt_tokens", type=int, default=16)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    print(f"torch {torch.__version__}  gpu = {torch.cuda.get_device_name(0)}")
    tok = AutoTokenizer.from_pretrained(a.dense)
    ids = torch.randint(0, 10000, (1, a.prompt_tokens), device="cuda")

    res = {}

    # -- dense reference: the original model on cuBLAS --
    torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(
        a.dense, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    res["dense"] = run(m, ids, a.new_tokens)
    res["dense"]["weights_gb"] = sum(p.numel() * p.element_size()
                                     for p in m.parameters()) / 1e9
    print(f"dense  : {res['dense']}")
    del m; gc.collect(); torch.cuda.empty_cache()

    # -- sparse on MACKO --
    torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(
        a.sparse, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    t0 = time.perf_counter()
    n = compress_model(m)
    print(f"compressed {n} Linears in {time.perf_counter() - t0:.1f}s")
    res["macko"] = run(m, ids, a.new_tokens)
    res["macko"]["weights_gb"] = sum(
        b.numel() * b.element_size() for b in m.buffers()) / 1e9
    print(f"macko  : {res['macko']}")
    del m; gc.collect(); torch.cuda.empty_cache()

    d, s = res["dense"], res["macko"]
    print(f"\n{'':<22}{'dense':>12}{'MACKO 80%':>12}{'ratio':>10}")
    print("-" * 56)
    for k, lbl, inv in [("decode_ms_median", "decode ms/token", True),
                        ("decode_tok_s", "decode tokens/s", False),
                        ("peak_gb", "peak GPU GB", True),
                        ("prefill_ms", "prefill ms", True)]:
        x, y = d[k], s[k]
        r = (x / y) if inv else (y / x)
        print(f"{lbl:<22}{x:12.2f}{y:12.2f}{r:9.3f}x")
    print("\n(>1 means the sparse model on MACKO is better)")
    print(f"decode spread dense p10/p90 {d['decode_ms_p10']:.2f}/{d['decode_ms_p90']:.2f} ms, "
          f"macko {s['decode_ms_p10']:.2f}/{s['decode_ms_p90']:.2f} ms")
    print("reference, SpMV kernel alone at density 0.20: 2.50x geometric mean")

    # The point of recording every step: show whether the gain holds up as the
    # KV cache grows. Buckets of 512 tokens, median within each.
    def med(v):
        v = sorted(v); return v[len(v) // 2]
    B = 512
    n = min(len(d["per_token_ms"]), len(s["per_token_ms"]))
    print(f"\n{'tokens generated':<20}{'dense ms':>10}{'macko ms':>10}{'speedup':>10}")
    print("-" * 50)
    for lo in range(0, n, B):
        hi = min(lo + B, n)
        if hi - lo < B // 4:
            break
        dm, sm = med(d["per_token_ms"][lo:hi]), med(s["per_token_ms"][lo:hi])
        print(f"{f'{lo}-{hi}':<20}{dm:10.2f}{sm:10.2f}{dm/sm:9.3f}x")
    print("\nIf the ratio falls with position, attention over the growing KV cache")
    print("is diluting a Linear-only speedup. If it rises, the fixed per-token")
    print("Python cost of MackoLayer was what held the early buckets back.")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
