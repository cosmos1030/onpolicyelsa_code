"""4B dense, 1.7B dense and 4B-sparse+MACKO, all in one job.

Run as three separate jobs these can land on three different cards, and then
none of the ratios mean anything -- the 4B dense reference this is measured
against was taken on an L40S. Measuring them in one allocation makes the
comparison self-contained whatever card SLURM hands out.

The question is the one a reviewer asks first: if a 4B model has to be pruned
to 80% to get fast, why not just run the 1.7B? Accuracy already favours the
1.7B (58.42 vs SCOUT-4B-s70's 45.58), so speed and memory are what is left to
argue with.

Usage: bench_vllm_three.py --m4 <dir> --m17 <dir> --sparse <dir> [--label s70]
"""
import argparse, gc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def measure(llm, lengths, tag):
    from vllm import SamplingParams
    out = {}
    pid = list(range(16))
    for L in lengths:
        llm.generate(prompt_token_ids=[pid],
                     sampling_params=SamplingParams(temperature=0.0, max_tokens=8,
                                                    ignore_eos=True))
        sp = SamplingParams(temperature=0.0, max_tokens=L, ignore_eos=True)
        t0 = time.perf_counter()
        r = llm.generate(prompt_token_ids=[pid], sampling_params=sp)
        dt = time.perf_counter() - t0
        n = len(r[0].outputs[0].token_ids)
        out[L] = dict(total_sec=dt, tokens=n, ms_per_token=dt * 1e3 / n)
        print(f"  [{tag}] L={L:<6} {n} tok in {dt:6.2f}s -> {dt*1e3/n:6.2f} ms/token",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m4", required=True)
    ap.add_argument("--m17", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--label", default="sparse")
    ap.add_argument("--lengths", default="512,2048,4096,8192")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]

    import torch
    from vllm import LLM
    import macko_linear      # noqa: F401 -- registers "macko"

    gpu = torch.cuda.get_device_name(0)
    print(f"gpu = {gpu}   (all three models measured on THIS card)")

    res, order = {}, [("4B dense", a.m4, None), ("1.7B dense", a.m17, None),
                      (f"4B {a.label}", a.sparse, "macko")]
    for tag, path, quant in order:
        kw = dict(model=path, dtype="float16", max_model_len=max(lengths) + 64,
                  gpu_memory_utilization=0.85, enforce_eager=False,
                  max_num_seqs=1, disable_log_stats=True)
        if quant:
            kw["quantization"] = quant
        llm = LLM(**kw)
        print(f"--- {tag} ---", flush=True)
        res[tag] = measure(llm, lengths, tag)
        del llm; gc.collect(); torch.cuda.empty_cache()

    d4, d17 = res["4B dense"], res["1.7B dense"]
    sp = res[f"4B {a.label}"]
    print(f"\n{'L':>6}{'4B dense':>11}{'1.7B dense':>13}{f'4B {a.label}':>13}"
          f"{'vs 4Bdense':>12}{'vs 1.7B':>10}")
    print("-" * 65)
    for L in lengths:
        x, y, z = d4[L]["ms_per_token"], d17[L]["ms_per_token"], sp[L]["ms_per_token"]
        print(f"{L:>6}{x:11.2f}{y:13.2f}{z:13.2f}{x/z:11.3f}x{y/z:9.3f}x")
    print("\n(>1 means the sparse 4B is faster than that column's model)")
    print("avg5 for context: 4B dense 68.62, 1.7B dense 58.42, SCOUT 4B s70 45.58")

    if a.out:
        json.dump(dict(gpu=gpu, **{k: v for k, v in res.items()}), open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
