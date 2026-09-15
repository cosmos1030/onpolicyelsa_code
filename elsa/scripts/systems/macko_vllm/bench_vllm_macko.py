"""Batch-1 decode latency in vLLM: dense vs 80%-sparse on MACKO.

This is the comparison the earlier HF-based one could not make honestly. Both
sides now run under the same engine, the same PagedAttention, the same CUDA
graphs -- the only difference is which kernel the Linear layers use. The dense
reference here is the one a person would actually deploy (13.7 ms/token at 8192
on L40S), not HF generate() (32.95).

--max-num-seqs 1 is not a benchmarking convenience, it is the regime MACKO
addresses: an SpMV kernel multiplies by a vector, so a decode step with N
scheduled sequences needs N launches and the dense kernel wins immediately.
What is being claimed is single-stream latency, which is what a reasoning model
emitting 8000 tokens actually costs a waiting user.

Usage: bench_vllm_macko.py --dense <dir> --sparse <dir>
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def measure(llm, lengths, tag):
    from vllm import SamplingParams
    out = {}
    prompt_ids = list(range(16))
    for L in lengths:
        llm.generate(prompt_token_ids=[prompt_ids],
                     sampling_params=SamplingParams(temperature=0.0,
                                                    max_tokens=8,
                                                    ignore_eos=True))
        sp = SamplingParams(temperature=0.0, max_tokens=L, ignore_eos=True)
        t0 = time.perf_counter()
        r = llm.generate(prompt_token_ids=[prompt_ids], sampling_params=sp)
        dt = time.perf_counter() - t0
        n = len(r[0].outputs[0].token_ids)
        out[L] = dict(total_sec=dt, tokens=n, ms_per_token=dt * 1e3 / n,
                      tok_s=n / dt)
        print(f"  [{tag}] L={L:<6} {n} tok in {dt:6.2f}s -> "
              f"{dt*1e3/n:6.2f} ms/token ({n/dt:6.1f} tok/s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--lengths", default="512,2048,4096,8192")
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]

    import torch
    from vllm import LLM
    import macko_linear      # noqa: F401  -- registers the "macko" method

    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    print(f"gpu = {torch.cuda.get_device_name(0)}")
    print(f"'macko' registered: {'macko' in QUANTIZATION_METHODS}")

    res = {}
    for tag, path, quant in (("dense", a.dense, None),
                             ("macko_s80", a.sparse, "macko")):
        kw = dict(model=path, dtype="float16", max_model_len=max(lengths) + 64,
                  gpu_memory_utilization=a.gpu_util, enforce_eager=False,
                  max_num_seqs=1, disable_log_stats=True)
        if quant:
            kw["quantization"] = quant
        t0 = time.perf_counter()
        llm = LLM(**kw)
        print(f"--- {tag} --- engine up in {time.perf_counter()-t0:.0f}s", flush=True)
        # How many Linears actually took the MACKO path, and at what density.
        if quant:
            n_c = n_s = 0
            dens = []
            for m in llm.llm_engine.model_executor.driver_worker.model_runner.model.modules():
                if hasattr(m, "macko_density"):
                    n_s += 1
                    dens.append(m.macko_density)
                    if getattr(m, "macko_compressed", None) is not None:
                        n_c += 1
            if dens:
                print(f"    Linears seen {n_s}, compressed {n_c}, "
                      f"density min {min(dens):.3f} max {max(dens):.3f}")
        res[tag] = measure(llm, lengths, tag)
        del llm
        import gc; gc.collect(); torch.cuda.empty_cache()

    d, s = res["dense"], res["macko_s80"]
    print(f"\n{'output length':<16}{'dense ms/tok':>15}{'macko ms/tok':>15}{'speedup':>10}")
    print("-" * 56)
    for L in lengths:
        x, y = d[L]["ms_per_token"], s[L]["ms_per_token"]
        print(f"{L:<16}{x:15.2f}{y:15.2f}{x/y:9.3f}x")
    print("\n(>1 means the sparse model on MACKO is faster, same engine both sides)")
    print("for reference, outside vLLM on the same card: SpMV kernel 2.58x,")
    print("HF generate()+CUDA graph 1.26x at 8192, HF dense 32.95 ms/token")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
