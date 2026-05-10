"""
Generate teacher CoT completions using vLLM offline inference.
Saves to JSONL with 'text' field in MathCotKDDataset format:
  "problem\n\n<think>CoT</think>answer"
"""
import argparse
import json
import re
import os
from vllm import LLM, SamplingParams

BOS = "<｜begin▁of▁sentence｜>"
USER_TAG = "<｜User｜>"
ASST_TAG = "<｜Assistant｜>"


def extract_problem(prompt: str) -> str:
    """Strip chat template from prompt to get raw problem text."""
    text = prompt
    if BOS in text:
        text = text[text.index(BOS) + len(BOS):]
    if USER_TAG in text:
        text = text[text.index(USER_TAG) + len(USER_TAG):]
    # Remove trailing <|Assistant|><think>\n
    for suffix in [ASST_TAG + "<think>\n", ASST_TAG + "<think>", ASST_TAG]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="math_220k_prompts.jsonl")
    parser.add_argument("--output", required=True, help="output jsonl path")
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # Load prompts
    records = []
    with open(args.input) as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
            if len(records) >= args.max_samples:
                break
    print(f"Loaded {len(records)} prompts")

    prompts = [r["prompt"] for r in records]

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=10240,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    print(f"Generating {len(prompts)} completions...")
    outputs = llm.generate(prompts, sampling_params)

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    written = 0
    with open(args.output, "w") as f:
        for rec, out in zip(records, outputs):
            generated = out.outputs[0].text  # continuation after <think>
            problem = extract_problem(rec["prompt"])
            # Reconstruct in MathCotKDDataset format
            text = problem + "\n\n<think>" + generated
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Saved {written} samples to {args.output}")


if __name__ == "__main__":
    main()
