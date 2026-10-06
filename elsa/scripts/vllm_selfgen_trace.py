"""
Fast vLLM-based replacement for RAC's grpo.py --trace_only self-gen trace
generation. The old path used plain HF generate() (no batching/paging), which
crawls at ~10-27 tok/s per sequence -- for a 2,000,000-token budget with
num_generations=2 that took the better part of a day. vLLM's continuous
batching does the exact same thing (dense model, native chat template, no
system_prompt override, thinking mode on by default) at a small fraction of
the cost.

Also drops the num_generations=2 + dedup-to-102-unique-problems dance the v3
1.7B pipeline needed: since we sample each of n_prompts DIFFERENT prompts
exactly ONCE (n=1), the output is already "1 answer per problem" -- no
dedup step needed before feeding into build_selfgen_ot3_fineweb_dataset.py.

Usage:
    python vllm_selfgen_trace.py \
        --model_path <Qwen3 model dir> \
        --prompt_path /home1/doyoonkim/projects/elsa/data/ot3_prompts_2000_qwen3.jsonl \
        --out_path <trace jsonl, prompt+completion columns> \
        --n_prompts 150 --seed 42 --max_tokens 8192 --temperature 0.7
"""
import argparse
import json
import random

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prompt_path", default="/home1/doyoonkim/projects/elsa/data/ot3_prompts_2000_qwen3.jsonl")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--n_prompts", type=int, default=150,
                     help="Number of DISTINCT prompts to sample (>102 for buffer against short/failed completions)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gpu_util", type=float, default=0.9)
    ap.add_argument("--tp_size", type=int, default=1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    rows = [json.loads(l) for l in open(args.prompt_path)]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.n_prompts]
    print(f"Sampled {len(rows)} distinct prompts (seed={args.seed})", flush=True)

    chat_prompts = []
    for r in rows:
        messages = [{"role": "user", "content": r["prompt"]}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(text)

    llm = LLM(model=args.model_path, trust_remote_code=True, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_util, tensor_parallel_size=args.tp_size,
              max_model_len=args.max_tokens + 1024, seed=args.seed)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95, top_k=20,
                         max_tokens=args.max_tokens, n=1)

    print("Generating...", flush=True)
    outputs = llm.generate(chat_prompts, sp)

    n_written = 0
    with open(args.out_path, "w") as f:
        for prompt_text, out in zip(chat_prompts, outputs):
            completion = out.outputs[0].text
            if not completion.strip():
                continue
            f.write(json.dumps({"prompt": prompt_text, "completion": completion}, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"Wrote {n_written}/{len(rows)} rows to {args.out_path}", flush=True)


if __name__ == "__main__":
    main()
