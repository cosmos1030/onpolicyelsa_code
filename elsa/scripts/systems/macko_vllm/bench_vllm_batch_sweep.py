"""What batch-1 costs: dense vLLM throughput as concurrency rises.

The MACKO claim is a single-stream latency claim, and it has to be, because an
SpMV kernel multiplies by a vector -- a decode step with N scheduled sequences
needs N launches, so the dense kernel wins the moment concurrency exceeds one.
The integration in macko_linear.py makes that explicit: apply() falls back to
F.linear whenever the token count is above 1.

Which means a reviewer asking "why only batch 1?" deserves the number for what
that choice gives up. This measures the dense model alone, at several
concurrency levels, reporting both quantities that matter:

  per-seq ms/token   what one waiting user experiences
  total tokens/s     what the server delivers across all users

Latency degrades slowly with batch while throughput scales nearly linearly,
which is the entire reason vLLM batches. Nothing here is a MACKO measurement;
it is the cost of the regime MACKO is restricted to.

Usage: bench_vllm_batch_sweep.py --model <dir> [--batches 1,4,8,32]
"""
import argparse, json, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batches", default="1,4,8,32")
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    batches = [int(x) for x in a.batches.split(",")]

    import torch
    from vllm import LLM, SamplingParams

    print(f"gpu = {torch.cuda.get_device_name(0)}  length={a.length}")
    llm = LLM(model=a.model, dtype="float16",
              max_model_len=a.length + 64, gpu_memory_utilization=a.gpu_util,
              enforce_eager=False, max_num_seqs=max(batches),
              disable_log_stats=True)

    res = {}
    for B in batches:
        # Distinct prompts so nothing dedupes or shares a prefix cache entry.
        prompts = [list(range(16 + i, 32 + i)) for i in range(B)]
        sp = SamplingParams(temperature=0.0, max_tokens=a.length, ignore_eos=True)
        llm.generate(prompt_token_ids=prompts[:1],
                     sampling_params=SamplingParams(temperature=0.0,
                                                    max_tokens=8, ignore_eos=True))
        t0 = time.perf_counter()
        outs = llm.generate(prompt_token_ids=prompts, sampling_params=sp)
        dt = time.perf_counter() - t0
        tot = sum(len(o.outputs[0].token_ids) for o in outs)
        res[B] = dict(batch=B, wall_sec=dt, total_tokens=tot,
                      throughput_tok_s=tot / dt,
                      per_seq_ms_per_token=dt * 1e3 / (tot / B))
        print(f"  B={B:<4} {dt:7.2f}s  {tot:6d} tok  "
              f"throughput {tot/dt:8.1f} tok/s  "
              f"per-seq {dt*1e3/(tot/B):6.2f} ms/token", flush=True)

    base = res[batches[0]]
    print(f"\n{'batch':<8}{'per-seq ms/tok':>16}{'vs B=1':>9}"
          f"{'throughput tok/s':>19}{'vs B=1':>9}")
    print("-" * 61)
    for B in batches:
        r = res[B]
        print(f"{B:<8}{r['per_seq_ms_per_token']:16.2f}"
              f"{r['per_seq_ms_per_token']/base['per_seq_ms_per_token']:8.2f}x"
              f"{r['throughput_tok_s']:19.1f}"
              f"{r['throughput_tok_s']/base['throughput_tok_s']:8.2f}x")
    print("\nMACKO is available only in the B=1 row: its apply() falls back to")
    print("the dense kernel above one token per step, so every other row is")
    print("what restricting to B=1 forfeits.")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
