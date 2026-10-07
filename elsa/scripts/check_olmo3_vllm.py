"""Check that vLLM (with vllm_olmo3_plugin) matches HF transformers on OLMo 3.

Feeds one prompt longer than the 4096-token sliding window to both and
compares per-position log-probs of the actual next token. A wrong sliding
window or rope scaling shows up as a jump in the error after position 4096.

    CUDA_VISIBLE_DEVICES=0 python elsa/scripts/check_olmo3_vllm.py --model allenai/Olmo-3-7B-Think
"""
import argparse
import gc

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Think")
    ap.add_argument("--n_tokens", type=int, default=6000)
    ap.add_argument("--data", default="/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl")
    args = ap.parse_args()

    import json
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    text = ""
    with open(args.data) as f:
        for line in f:
            text += json.loads(line)["text"]
            if len(tok(text, add_special_tokens=False).input_ids) > args.n_tokens:
                break
    ids = tok(text, add_special_tokens=False).input_ids[:args.n_tokens]

    # --- HF ---
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    with torch.no_grad():
        logits = model(torch.tensor([ids], device="cuda")).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    hf = lp[torch.arange(len(ids) - 1), torch.tensor(ids[1:])].cpu()
    del model, logits, lp
    gc.collect(); torch.cuda.empty_cache()

    # --- vLLM (V0, same engine the training sidecar uses) ---
    import os
    os.environ.setdefault("VLLM_USE_V1", "0")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.5,
              max_model_len=args.n_tokens + 16, enforce_eager=True)
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                       SamplingParams(max_tokens=1, prompt_logprobs=0))[0]
    vl = torch.tensor([list(d.values())[0].logprob if d else float("nan")
                       for d in out.prompt_logprobs[1:]])
    # prompt_logprobs[i] is the logprob of token i given tokens < i
    assert len(vl) == len(hf), (len(vl), len(hf))

    err = (vl - hf).abs()
    for lo, hi in [(0, 1024), (1024, 4096), (4096, 5000), (5000, len(hf))]:
        e = err[lo:hi]
        print(f"positions {lo:5d}-{hi:5d}: mean |dlogp| {e.mean():.4f}  max {e.max():.4f}  "
              f"(HF mean logp {hf[lo:hi].mean():.3f})")
    ok = err[4096:].mean() < 3 * err[:4096].mean() + 0.02
    print("RESULT:", "PASS" if ok else "FAIL (error jumps past the sliding window)")


if __name__ == "__main__":
    main()
