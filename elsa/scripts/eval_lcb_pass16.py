"""
Evaluate pass@1:16 on LiveCodeBench (lcb:codegeneration, v4_v5 subset)
using vLLM with n=16 samples per problem.

Usage:
    python scripts/eval_lcb_pass16.py --model deepseek-1.5b
    python scripts/eval_lcb_pass16.py --model qwen3-1.7b
    python scripts/eval_lcb_pass16.py --model /path/to/custom/model
"""

import argparse
import json
import os
import sys
import time

MODEL_ALIASES = {
    "deepseek-1.5b": "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
    "qwen3-1.7b":    "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}

def prepare_prompt(line):
    query = "You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n"
    query += f"Question: {line['question_content']}\n\n"
    if starter_code := line.get("starter_code") or "":
        query += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
        query += f"```python\n{starter_code}\n```\n\n"
    else:
        query += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
        query += "```python\n# YOUR CODE HERE\n```\n\n"
    return query


def build_samples_and_prompts(dataset):
    from lighteval.tasks.tasks.lcb.codegen_metrics import translate_private_test_cases
    samples, prompts = [], []
    for line in dataset:
        public = json.loads(line["public_test_cases"])
        private = translate_private_test_cases(line["private_test_cases"])
        inputs  = [t["input"]  for t in public + private]
        outputs = [t["output"] for t in public + private]
        fn_name = json.loads(line["metadata"]).get("func_name", None)
        samples.append({"input_output": json.dumps({"inputs": inputs, "outputs": outputs, "fn_name": fn_name})})
        prompts.append(prepare_prompt(line))
    return samples, prompts


def generate_with_vllm(model_path, prompts, n, temperature, max_tokens, gpu_util):
    from vllm import LLM, SamplingParams
    print(f"[vllm] Loading {model_path}", flush=True)
    llm = LLM(model=model_path, gpu_memory_utilization=gpu_util, max_model_len=32768)
    params = SamplingParams(n=n, temperature=temperature, top_p=0.95, max_tokens=max_tokens, stop=["<|im_end|>", "<|endoftext|>"])
    print(f"[vllm] Generating {n} samples × {len(prompts)} problems …", flush=True)
    t0 = time.time()
    outputs = llm.generate(prompts, params)
    print(f"[vllm] Done in {time.time()-t0:.1f}s", flush=True)
    # generations[i] = list of n raw text outputs for problem i
    generations = [[o.text for o in req.outputs] for req in outputs]
    return generations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Alias (deepseek-1.5b / qwen3-1.7b) or full path")
    parser.add_argument("--n", type=int, default=16, help="Samples per problem")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--gpu_util", type=float, default=0.85)
    parser.add_argument("--num_process_evaluate", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit problems (smoke test)")
    parser.add_argument("--out", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    model_path = MODEL_ALIASES.get(args.model, args.model)
    if not os.path.exists(model_path):
        print(f"ERROR: model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    from datasets import load_dataset
    print("[data] Loading lighteval/code_generation_lite v4_v5 …", flush=True)
    ds = load_dataset("lighteval/code_generation_lite", "v4_v5", split="test")
    if args.max_samples:
        ds = ds.select(range(args.max_samples))
    print(f"[data] {len(ds)} problems", flush=True)

    samples, prompts = build_samples_and_prompts(ds)

    generations_raw = generate_with_vllm(
        model_path, prompts, args.n, args.temperature, args.max_tokens, args.gpu_util
    )

    from lighteval.tasks.tasks.lcb.codegen_metrics import codegen_metrics, extract_code
    print("[eval] Extracting code and running codegen_metrics …", flush=True)
    # generations must be list[list[str]] — one inner list per problem, each element is one code snippet
    generations_code = [[extract_code(text) for text in prob_gens] for prob_gens in generations_raw]

    k_list = [k for k in [1, args.n] if k <= args.n]
    metrics, results = codegen_metrics(
        samples,
        generations_code,
        k_list=k_list,
        num_process_evaluate=args.num_process_evaluate,
    )

    print("\n===== RESULTS =====")
    model_name = args.model
    for k in k_list:
        key = f"pass@{k}"
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    print("===================\n")

    if args.out:
        out = {
            "model": model_path,
            "n": args.n,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "num_problems": len(ds),
            "metrics": {k: v for k, v in metrics.items() if k != "detail"},
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[out] Saved to {args.out}")

    import wandb
    if wandb.run is not None:
        log = {f"lcb_pass@{k}": metrics[f"pass@{k}"] for k in k_list if f"pass@{k}" in metrics}
        log["model"] = model_name
        wandb.log(log)


if __name__ == "__main__":
    main()
