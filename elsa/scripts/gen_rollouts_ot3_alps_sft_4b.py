"""Generate rollouts from the ALPS one-shot + sparse SFT (NTP, fixed mask) Qwen3-4B
s70pct checkpoint on OpenThoughts3-1.2M prompts, and push the resulting
(prompt, completion) dataset to HuggingFace.

Same recipe as gen_rollouts_ot3_pruned4b.py, just pointed at the new model
(job 692163, uploaded to https://huggingface.co/cosmos1030/alps-sparse-sft-qwen3-4b-s70pct).
thinking mode is ON (apply_chat_template's enable_thinking defaults to True
for Qwen3 when not explicitly passed).

Usage: python gen_rollouts_ot3_alps_sft_4b.py
"""
import os
import json

from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
from vllm import LLM, SamplingParams

MODEL_PATH = "/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_20260805_111647"
N_PROMPTS = 20000
MAX_NEW_TOKENS = 2048
MAX_PROMPT_LEN = 1024
TEMPERATURE = 0.6
TOP_P = 0.95
OUT_REPO = "cosmos1030/qwen3-4b-alps-sparse-sft-s70pct-ot3-rollouts"
OUT_JSONL = "/home1/doyoonkim/projects/elsa/data/ot3_rollouts_alps_sft_4b_s70_20k.jsonl"

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

    print(f"Collected {len(prompts)} prompts within {MAX_PROMPT_LEN} tokens. Generating...", flush=True)

    params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)
    outputs = llm.generate(prompts, params)

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    records = []
    with open(OUT_JSONL, "w") as f:
        for raw_prompt, out in zip(raw_prompts, outputs):
            completion = out.outputs[0].text
            rec = {"prompt": raw_prompt, "completion": completion}
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} rollouts to {OUT_JSONL}", flush=True)

    hf_ds = Dataset.from_list(records)
    token = os.environ.get("HF_TOKEN") or open(os.path.expanduser("~/.hf_token")).read().strip()
    api = HfApi(token=token)
    api.create_repo(repo_id=OUT_REPO, repo_type="dataset", exist_ok=True)
    hf_ds.push_to_hub(OUT_REPO, token=token)
    print(f"Pushed dataset to https://huggingface.co/datasets/{OUT_REPO}", flush=True)


if __name__ == "__main__":
    main()
