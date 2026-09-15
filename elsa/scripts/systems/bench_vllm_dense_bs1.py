"""Is the dense baseline in the MACKO comparison a strawman?

Everything measured so far runs both sides through HF generate(). That keeps
the comparison internally fair -- MACKO replaces nn.Linear modules and has no
vLLM integration, so putting only the dense side on vLLM would measure serving
stacks rather than models. But it leaves an obvious objection: if HF generate()
is simply a slow way to run a dense model, then "1.26x over dense" is 1.26x
over a handicapped reference and means nothing.

So this measures the same dense model, same card, same batch size 1, same fixed
output length, on vLLM. The number to compare against is HF generate() + CUDA
graph, which came in at 32.95 ms/token at 8192 KV on L40S.

  vLLM close to that  -> the baseline is honest and the MACKO result stands
  vLLM much faster    -> the claim has to be narrowed to "vs HF generate()",
                         or MACKO has to go into vLLM after all

Batch 1 and ignore_eos on purpose: this is a latency measurement, not a
throughput one, and MACKO's SpMV kernel only addresses batch 1 anyway.

Usage: bench_vllm_dense_bs1.py --model <dir> [--lengths 512,2048,4096,8192]
"""
import argparse, json, os, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="512,2048,4096,8192")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]

    from vllm import LLM, SamplingParams
    import torch

    print(f"gpu = {torch.cuda.get_device_name(0)}")
    llm = LLM(model=a.model, dtype="float16", max_model_len=max(lengths) + 64,
              gpu_memory_utilization=0.85, enforce_eager=False,
              disable_log_stats=True)
    print("engine up (CUDA graphs on: enforce_eager=False)")

    res = {}
    for L in lengths:
        # One request, L new tokens, EOS ignored so every run generates exactly
        # L -- the same protocol as the HF-side measurement.
        sp = SamplingParams(temperature=0.0, max_tokens=L, ignore_eos=True)
        prompt_ids = list(range(16))

        llm.generate(prompt_token_ids=[prompt_ids],
                     sampling_params=SamplingParams(temperature=0.0, max_tokens=8,
                                                    ignore_eos=True))  # warm
        t0 = time.perf_counter()
        out = llm.generate(prompt_token_ids=[prompt_ids], sampling_params=sp)
        dt = time.perf_counter() - t0

        n = len(out[0].outputs[0].token_ids)
        # Whole-request time over tokens generated. This folds prefill (16
        # tokens, negligible) into the average, and averages over a KV cache
        # that grows from 16 to L -- so it is not the same quantity as a
        # single decode step at a pinned KV length. Reported as both.
        res[L] = dict(total_sec=dt, tokens=n, ms_per_token=dt * 1e3 / n,
                      tok_s=n / dt)
        print(f"  L={L:<6} {n} tokens in {dt:6.2f}s -> "
              f"{dt*1e3/n:6.2f} ms/token  ({n/dt:6.1f} tok/s)")

    print(f"\n{'output length':<16}{'vLLM ms/tok':>14}{'HF+graph ms/tok':>18}{'ratio':>9}")
    print("-" * 57)
    # HF generate() + CUDA graph, dense, L40S, measured at a PINNED KV length
    # (bench_macko_graph_e2e.py, job 931505). vLLM's number averages over a
    # growing cache, so the honest comparison is against the midpoint of the
    # HF curve, not its endpoint -- both are printed.
    HF_PINNED = {512: 15.35, 2048: 18.07, 4096: 22.89, 8192: 32.95}
    for L in lengths:
        hf = HF_PINNED.get(L)
        if hf is None:
            continue
        v = res[L]["ms_per_token"]
        print(f"{L:<16}{v:14.2f}{hf:18.2f}{hf/v:8.3f}x")
    print("\n>1 means vLLM is faster than HF generate()+CUDA graph at that length.")
    print("Note the two are not measuring the same thing: vLLM's value averages")
    print("over a KV cache growing 16->L, while the HF value is one decode step")
    print("at a cache already holding L. vLLM should therefore look better even")
    print("if the per-step cost is identical; treat a ratio near or below 1 as")
    print("strong evidence the HF baseline is not handicapped.")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
