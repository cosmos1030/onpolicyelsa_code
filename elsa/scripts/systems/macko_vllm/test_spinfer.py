"""SpInfer plug-in checks on a real SCOUT checkpoint:
 1. op level: spmm vs F.linear on every fused shape, n in {1,3,8,20,64,128,300,2048}, eager + CUDA graph
 2. vLLM level: greedy outputs of dense-weight vLLM vs SPARSE_BACKEND=spinfer on the same sparse model
    (must match token for token up to fp16 accumulation-order drift), batch 8.
    python test_spinfer.py --model <dir> [--vllm]
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open


def op_test(model):
    import spinfer_ops
    spinfer_ops.load()
    idx = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]

    def get(n):
        with safe_open(os.path.join(model, idx[n]), "pt") as f:
            return f.get_tensor(n)
    p = "model.layers.10."
    W = dict(qkv=torch.cat([get(p + f"self_attn.{k}_proj.weight") for k in "qkv"]),
             o=get(p + "self_attn.o_proj.weight"),
             gate_up=torch.cat([get(p + "mlp.gate_proj.weight"), get(p + "mlp.up_proj.weight")]),
             down=get(p + "mlp.down_proj.weight"))
    ok = True
    for name, w in W.items():
        w = w.cuda().half()
        M, K = w.shape
        t0 = time.time()
        c = spinfer_ops.compress(w)
        torch.cuda.synchronize()
        sk = spinfer_ops.tune_split_k(c, M, K)
        csz = sum(t.numel() * t.element_size() for t in c)
        print(f"{name} M{M} K{K} dens {(w != 0).float().mean():.3f} compress {time.time()-t0:.2f}s "
              f"size {csz/w.numel()/2:.3f}x dense  split_k {sk}", flush=True)
        for n in (1, 3, 8, 20, 64, 128, 300, 2048):
            x = torch.randn(n, K, device="cuda", dtype=torch.float16)
            ref = F.linear(x.float(), w.float())
            y = spinfer_ops.spmm(x, c, M, sk.get(spinfer_ops.pad_n(n), 1)).float()
            err = ((y - ref).abs().max() / ref.abs().max()).item()
            # CUDA graph capture of the same call
            xs = x.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                spinfer_ops.spmm(xs, c, M, sk.get(spinfer_ops.pad_n(n), 1))
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                yg = spinfer_ops.spmm(xs, c, M, sk.get(spinfer_ops.pad_n(n), 1))
            xs.copy_(torch.randn_like(xs))
            g.replay()
            torch.cuda.synchronize()
            refg = F.linear(xs.float(), w.float())
            errg = ((yg.float() - refg).abs().max() / refg.abs().max()).item()
            flag = "OK" if max(err, errg) < 5e-3 else "BAD"
            ok &= flag == "OK"
            print(f"   n={n:5d} rel_err eager {err:.2e} graph {errg:.2e} {flag}", flush=True)
    return ok


def vllm_test(model, chat_tok, be, out):
    import macko_linear  # noqa: F401  registers "macko" (backend from SPARSE_BACKEND)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(chat_tok)
    qs = ["What is 17*23?", "Name three prime numbers above 100.", "Solve x^2-5x+6=0.",
          "What is the capital of France?", "Compute 2^10.", "Is 91 prime?",
          "Simplify (a+b)^2.", "What is 1/3 + 1/6?"]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                       add_generation_prompt=True) for q in qs]
    sp = SamplingParams(temperature=0.0, max_tokens=256)
    kw = dict(model=model, dtype="float16", max_model_len=4096, gpu_memory_utilization=0.85,
              max_num_seqs=8, enforce_eager=False, disable_log_stats=True)
    if be != "dense":
        kw["quantization"] = "macko"
    llm = LLM(**kw)
    llm.generate(prompts[:2], SamplingParams(temperature=0.0, max_tokens=8))   # warm-up
    t = time.time()
    o = llm.generate(prompts, sp)
    dt = time.time() - t
    ids = [list(x.outputs[0].token_ids) for x in o]
    ntok = sum(len(x) for x in ids)
    print(f"[{be}] {ntok} tokens in {dt:.1f}s = {ntok/dt:.1f} tok/s (batch 8, greedy)", flush=True)
    print("   sample:", o[0].outputs[0].text[:200].replace("\n", " "), flush=True)
    json.dump(ids, open(out, "w"))
    other = out.replace(f"_{be}.json", "_dense.json" if be != "dense" else "_spinfer.json")
    if os.path.exists(other):
        ref = json.load(open(other))
        agree = []
        for a_, b_ in zip(ref, ids):
            k = 0
            while k < min(len(a_), len(b_)) and a_[k] == b_[k]:
                k += 1
            agree.append(k)
        print("tokens identical before first divergence (of 256):", agree, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--chat_tokenizer", required=True)
    ap.add_argument("--vllm", default=None, help="dense|spinfer: run only the vLLM check")
    ap.add_argument("--out", default="/tmp/spinfer_vllm_ids")
    a = ap.parse_args()
    if a.vllm:
        vllm_test(a.model, a.chat_tokenizer, a.vllm, f"{a.out}_{a.vllm}.json")
    else:
        ok = op_test(a.model)
        print("OP TEST", "PASS" if ok else "FAIL", flush=True)
