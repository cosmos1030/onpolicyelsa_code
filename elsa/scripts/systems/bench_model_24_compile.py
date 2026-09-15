"""Does torch.compile rescue 2:4, or is the kernel simply not fast enough here?

The eager measurement was dominated by ~0.7ms of Python dispatch per Linear --
252 of them in Qwen3-4B, so ~176ms/token of pure wrapper. CUDA graphs removed
that and left decode at 0.76x and prefill at 0.94x, i.e. still slower than
dense. Two things are worth separating before calling 2:4 dead on this stack:

  1. Can torch.compile remove the dispatch cost the way a graph does, without
     a graph's fixed-shape constraint? If it cannot even trace the
     SparseSemiStructuredTensor subclass, that is itself the answer for anyone
     hoping to drop 2:4 into a normal serving path.
  2. With host cost gone from BOTH sides, is the remaining gap the kernel? The
     Linear benchmark says yes -- 0.58x at M=1, 1.19x at M=512 -- so the model
     numbers should land near those, and prefill should NOT be rescued past
     ~1.19x no matter what the compiler does.

Every mode is run on both execution paths, on the same weights.

Usage: bench_model_24_compile.py --model <dir> [--mode default|max-autotune]
"""
import argparse, gc, json, time
import torch
import torch.nn as nn
from torch.sparse import to_sparse_semi_structured
from transformers import AutoConfig, AutoModelForCausalLM

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def sparsify_(model):
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


def eager_ms(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", default="default")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(a.model)
    V = cfg.vocab_size
    print(f"torch {torch.__version__}  gpu={torch.cuda.get_device_name(0)}  "
          f"compile mode={a.mode}")

    results = {}
    for path_name in ("dense", "sparse"):
        m = AutoModelForCausalLM.from_pretrained(
            a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        n = sparsify_(m) if path_name == "sparse" else 0
        print(f"\n--- {path_name}" + (f" ({n} Linears compressed)" if n else "") + " ---")

        ids = torch.randint(0, V, (1, a.seqlen), device="cuda")
        r = {}
        r["prefill_eager"] = eager_ms(lambda: m(ids), iters=5)

        pref = m(ids, use_cache=True)
        cache = pref.past_key_values
        nxt = torch.randint(0, V, (1, 1), device="cuda")
        r["decode_eager"] = eager_ms(lambda: m(nxt, past_key_values=cache, use_cache=True),
                                     iters=20)

        # torch.compile. dynamic=False because both shapes here are fixed, and a
        # dynamic graph would reintroduce guard overhead this is meant to remove.
        try:
            t0 = time.perf_counter()
            mc = torch.compile(m, mode=a.mode, dynamic=False)
            mc(ids)                                  # triggers compilation
            r["compile_prefill_sec"] = time.perf_counter() - t0
            r["prefill_compiled"] = eager_ms(lambda: mc(ids), iters=5)
        except Exception as ex:
            r["prefill_compiled"] = float("nan")
            r["compile_err"] = f"{type(ex).__name__}: {ex}"[:200]
            print(f"  compile(prefill) FAILED: {r['compile_err']}")

        try:
            pref2 = m(ids, use_cache=True)
            cache2 = pref2.past_key_values
            t0 = time.perf_counter()
            mc2 = torch.compile(m, mode=a.mode, dynamic=False)
            mc2(nxt, past_key_values=cache2, use_cache=True)
            r["compile_decode_sec"] = time.perf_counter() - t0
            r["decode_compiled"] = eager_ms(
                lambda: mc2(nxt, past_key_values=cache2, use_cache=True), iters=20)
            del pref2, cache2
        except Exception as ex:
            r["decode_compiled"] = float("nan")
            r["compile_decode_err"] = f"{type(ex).__name__}: {ex}"[:200]
            print(f"  compile(decode) FAILED: {r['compile_decode_err']}")

        for k in ("prefill_eager", "prefill_compiled", "decode_eager", "decode_compiled"):
            print(f"  {k:<20} {r.get(k, float('nan')):9.2f} ms")
        results[path_name] = r
        del m, pref, cache
        gc.collect(); torch.cuda.empty_cache()
        torch.compiler.reset()

    d, s = results["dense"], results["sparse"]
    print(f"\n{'measurement':<22}{'dense':>10}{'2:4':>10}{'speedup':>10}")
    print("-" * 52)
    for k in ("prefill_eager", "prefill_compiled", "decode_eager", "decode_compiled"):
        x, y = d.get(k, float("nan")), s.get(k, float("nan"))
        sp = f"{x/y:.3f}x" if (y == y and y > 0 and x == x) else "n/a"
        print(f"{k:<22}{x:10.2f}{y:10.2f}{sp:>10}")
    print("\n(>1 means 2:4 execution beats dense execution of the same weights)")
    print("reference, Linear only under CUDA graph: M=1 0.58x, M=512 1.19x")

    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
