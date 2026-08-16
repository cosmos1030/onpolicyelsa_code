"""Generate long (max_new_tokens=8192) rollouts from a Qwen3-1.7B checkpoint on
OpenThoughts3-1.2M prompts, and push the resulting (prompt, completion) dataset
to HuggingFace -- for downstream rambling/sequence-characteristics analysis
comparing the NTP+KD-only (OPD ablation) model against the NTP+KD+OPKD model.
The earlier rollout dataset (qwen3-4b-alps-sparse-sft-s70pct-ot3-rollouts) used
max_new_tokens=2048, too short to see rambling/degenerate-length behavior.

Usage: python gen_rollouts_ot3_1.7b_long.py <model_path_or_hf_repo> <out_repo> <out_jsonl> [n_prompts]
"""
import os
import sys
import json

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from vllm import LLM, SamplingParams

MODEL_PATH = sys.argv[1]
OUT_REPO = sys.argv[2]
OUT_JSONL = sys.argv[3]
N_PROMPTS = int(sys.argv[4]) if len(sys.argv) > 4 else 500

MAX_NEW_TOKENS = 16384
MAX_PROMPT_LEN = 1024
TEMPERATURE = 0.6
TOP_P = 0.95

os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    print(f"Loading OpenThoughts3-1.2M (target {N_PROMPTS} prompts)...", flush=True)
    ds = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    ds = ds.shuffle(seed=42)

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        max_model_len=MAX_PROMPT_LEN + MAX_NEW_TOKENS,
    )
    tokenizer = llm.get_tokenizer()

    prompts, raw_prompts = [], []
    for row in ds:
        convs = row.get("conversations") or []
        human_turns = [c["value"] for c in convs if c.get("from") == "human"]
        if not human_turns:
            continue
        prompt_text = human_turns[0].strip()
        if not prompt_text:
            continue
        chat = [{"role": "user", "content": prompt_text}]
        rendered = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        n_tok = len(tokenizer(rendered).input_ids)
        if n_tok > MAX_PROMPT_LEN:
            continue
        prompts.append(rendered)
        raw_prompts.append(prompt_text)
        if len(prompts) >= N_PROMPTS:
            break

    print(f"Collected {len(prompts)} prompts within {MAX_PROMPT_LEN} tokens. Generating (max_new_tokens={MAX_NEW_TOKENS})...", flush=True)

    params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
    outputs = llm.generate(prompts, params)

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    records = []
    with open(OUT_JSONL, "w") as f:
        for raw_prompt, out in zip(raw_prompts, outputs):
            completion = out.outputs[0].text
            n_completion_tok = len(out.outputs[0].token_ids)
            finish_reason = out.outputs[0].finish_reason
            rec = {
                "prompt": raw_prompt,
                "completion": completion,
                "completion_tokens": n_completion_tok,
                "finish_reason": finish_reason,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    lens = [r["completion_tokens"] for r in records]
    n_truncated = sum(1 for r in records if r["finish_reason"] == "length")
    print(f"Wrote {len(records)} rollouts to {OUT_JSONL}", flush=True)
    print(f"completion_tokens: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f} "
          f"truncated(hit max_new_tokens)={n_truncated}/{len(records)}", flush=True)

    hf_ds = Dataset.from_list(records)
    token = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.hf_token")).read().strip()
    api = HfApi(token=token)
    api.create_repo(repo_id=OUT_REPO, repo_type="dataset", exist_ok=True)
    hf_ds.push_to_hub(OUT_REPO, token=token)
    print(f"Pushed dataset to https://huggingface.co/datasets/{OUT_REPO}", flush=True)


if __name__ == "__main__":
    main()
