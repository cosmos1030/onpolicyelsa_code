"""
Pre-generate chosen (teacher) continuations and save to disk cache.
Run this as a standalone script BEFORE GMP training so the vLLM process
exits cleanly and releases GPU memory before training starts.

Usage:
    python build_chosen_cache.py \
        --model_path <teacher_model_path> \
        --data_path <math_220k_unified.jsonl> \
        --total_steps 1024 \
        --batch_size 1 \
        --grad_accum 8 \
        --max_new_tokens 1024 \
        --temperature 0.7 \
        --cache_dir /home1/doyoonkim/projects/elsa/.cache/dpo_chosen \
        --gen_batch_size 64
"""
import argparse
import hashlib
import json
import logging
import os
import pickle
import random
import sys

import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# ── cache key (must match gmp_dpo._chosen_cache_key) ───────────────────────
def _chosen_cache_key(prompt_path: str, n_pairs: int, max_new_tokens: int, temperature: float) -> str:
    raw = f"{prompt_path}|{n_pairs}|{max_new_tokens}|{temperature:.4f}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── minimal dataset (same logic as MathCotKDDataset + NTPPromptWrapper) ────
class PromptDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_prompt_len=512, seed=42):
        random.seed(seed)
        with open(jsonl_path) as f:
            records = [json.loads(l) for l in f if l.strip()]
        random.shuffle(records)
        self.samples = []
        for rec in records:
            text = rec.get("text", "")
            if not text:
                continue
            idx = text.find("<think>")
            if idx == -1:
                continue
            prompt_text = text[:idx + len("<think>")]
            enc = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_prompt_len,
                return_tensors="pt",
                padding=False,
            )
            self.samples.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })
        logging.info(f"PromptDataset: {len(self.samples)} prompts loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(pad_id):
    def _collate(batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)
        ids_list, mask_list = [], []
        for x in batch:
            pad_len = max_len - x["input_ids"].shape[0]
            ids_list.append(torch.cat([
                torch.full((pad_len,), pad_id, dtype=torch.long),
                x["input_ids"],
            ]))
            mask_list.append(torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                x["attention_mask"],
            ]))
        return {
            "input_ids":      torch.stack(ids_list),
            "attention_mask": torch.stack(mask_list),
        }
    return _collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path",  required=True)
    parser.add_argument("--total_steps",   type=int,   default=1024)
    parser.add_argument("--batch_size",    type=int,   default=1)
    parser.add_argument("--grad_accum",    type=int,   default=8)
    parser.add_argument("--max_new_tokens",type=int,   default=1024)
    parser.add_argument("--temperature",   type=float, default=0.7)
    parser.add_argument("--max_prompt_len",type=int,   default=512)
    parser.add_argument("--cache_dir",     default="/home1/doyoonkim/projects/elsa/.cache/dpo_chosen")
    parser.add_argument("--gen_batch_size",type=int,   default=64)
    parser.add_argument("--gpu_mem",       type=float, default=0.85)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    n_pairs = args.total_steps * args.batch_size * args.grad_accum
    prompt_path_key = f"{args.data_path}|ntp_prompt_wrapper|gbs={args.batch_size*args.grad_accum}"
    key = _chosen_cache_key(prompt_path_key, n_pairs, args.max_new_tokens, args.temperature)
    cache_file = os.path.join(args.cache_dir, f"chosen_cache_{key}.pt")

    if os.path.exists(cache_file):
        existing = torch.load(cache_file, map_location="cpu")
        logging.info(f"Cache already exists: {cache_file} ({len(existing)} pairs). Nothing to do.")
        return

    logging.info(f"Building chosen cache: {n_pairs} pairs → {cache_file}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    pad_id = tok.pad_token_id or tok.eos_token_id

    ds = PromptDataset(args.data_path, tok, max_prompt_len=args.max_prompt_len, seed=args.seed)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn(pad_id))

    # collect first n_pairs prompts
    all_prompt_ids, all_prompt_masks = [], []
    for batch in loader:
        if len(all_prompt_ids) >= n_pairs:
            break
        all_prompt_ids.append(batch["input_ids"].squeeze(0))
        all_prompt_masks.append(batch["attention_mask"].squeeze(0))
    logging.info(f"Collected {len(all_prompt_ids)} prompts")

    # strip left padding
    token_ids_list = []
    for ids in all_prompt_ids:
        ids_list = ids.tolist()
        while ids_list and ids_list[0] == pad_id:
            ids_list.pop(0)
        token_ids_list.append(ids_list)

    os.environ["VLLM_USE_V1"] = "0"
    from vllm import LLM, SamplingParams
    logging.info(f"Initializing vLLM (gpu_memory_utilization={args.gpu_mem}) ...")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_new_tokens + 1024,
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    logging.info(f"Generating {n_pairs} continuations ...")
    outputs = llm.generate(prompt_token_ids=token_ids_list, sampling_params=sampling_params)

    cache = []
    for i, out in enumerate(outputs):
        cont_tokens = out.outputs[0].token_ids
        cont_ids  = torch.tensor(cont_tokens, dtype=torch.long).unsqueeze(0)
        cont_mask = torch.ones_like(cont_ids)
        cache.append({
            "prompt_input_ids":      all_prompt_ids[i].unsqueeze(0),
            "prompt_attention_mask": all_prompt_masks[i].unsqueeze(0),
            "chosen_input_ids":      cont_ids,
            "chosen_attention_mask": cont_mask,
        })

    os.makedirs(args.cache_dir, exist_ok=True)
    torch.save(cache, cache_file)
    logging.info(f"Saved {len(cache)} pairs → {cache_file}")


if __name__ == "__main__":
    main()
