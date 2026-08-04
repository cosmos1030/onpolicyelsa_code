"""Pre-generate the teacher-gen-KD (TGKD) chosen-continuation cache once,
offline, so real TR-GMP training jobs (gmp_teacher_gen_kd=true) load an
already-built cache instead of generating inline at startup.

Builds the exact same MixedTextDataset the real training job builds from
--data_path (same cache key -> shared, reused tokenization), then splits the
first n_pairs samples into two groups:
  - CoT samples (real problem prompt, prompt_ids non-empty): teacher
    generates a continuation via vLLM, forward-KL target = generated tokens.
  - Pretrain/FineWeb-Edu samples (prompt_ids empty by construction --
    MixedTextDataset has no "prompt" concept for plain pretrain text):
    no generation -- the raw text itself IS the target (matches how
    _kl_loss's prompt_len==0 case already treats these: forward-KL over the
    whole sequence, not a generated continuation).
Both groups are merged back into one cache file, in original order, saved
under the same cache_dir/key scheme lib/gmp_dpo.py's generate_chosen_cache
uses, so the real training job's cache lookup hits this file directly.
"""
import argparse
import os

import torch
from transformers import AutoTokenizer

from lib.gkd_admm_trainer import MixedTextDataset
from lib.gmp_dpo import _chosen_cache_key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_path", required=True,
                   help="Training data path (disjoint from OPD's --gmp_prompt_path).")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max_prompt_len", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--cache_dir", default="/home1/doyoonkim/projects/elsa/.cache/dpo_chosen")
    args = p.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = MixedTextDataset(
        jsonl_path=args.data_path,
        tokenizer=tokenizer,
        max_prompt_len=args.max_prompt_len,
        max_len=args.seqlen,
        append_eos=False,
        nsamples=None,
    )

    n_pairs = args.steps * args.batch_size * args.grad_accum
    gbs = args.batch_size * args.grad_accum
    prompt_path_key = f"{args.data_path}|ntp_prompt_wrapper|gbs={gbs}"
    key = _chosen_cache_key(prompt_path_key, n_pairs, args.max_new_tokens, args.temperature)
    cache_file = os.path.join(args.cache_dir, f"chosen_cache_{key}.pt")
    if os.path.exists(cache_file):
        print(f"[pregenerate_tgkd_cache] cache already exists at {cache_file}, nothing to do.")
        return

    samples = train_dataset.samples[:n_pairs]
    cot_idx, pretrain_idx = [], []
    for i, s in enumerate(samples):
        (cot_idx if s["prompt_ids"].shape[0] > 0 else pretrain_idx).append(i)

    print(f"[pregenerate_tgkd_cache] n_pairs={n_pairs} cot={len(cot_idx)} pretrain={len(pretrain_idx)} "
          f"prompt_path_key={prompt_path_key!r}")

    cache = [None] * len(samples)

    # Pretrain/FineWeb-Edu: no generation, raw text is the target.
    for i in pretrain_idx:
        s = samples[i]
        cache[i] = {
            "prompt_input_ids": s["prompt_ids"].unsqueeze(0),        # (1, 0)
            "prompt_attention_mask": s["prompt_mask"].unsqueeze(0),  # (1, 0)
            "chosen_input_ids": s["input_ids"].unsqueeze(0),
            "chosen_attention_mask": s["attention_mask"].unsqueeze(0),
        }

    # CoT: teacher generates a continuation from the real problem prompt via vLLM.
    if cot_idx:
        os.environ["VLLM_USE_V1"] = "0"
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=args.max_new_tokens + args.max_prompt_len + 64,
            enforce_eager=True,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=42,
        )
        token_ids_list = [samples[i]["prompt_ids"].tolist() for i in cot_idx]
        print(f"[pregenerate_tgkd_cache] [vLLM] generating {len(token_ids_list)} CoT continuations ...")
        outputs = llm.generate(prompt_token_ids=token_ids_list, sampling_params=sampling_params)
        for i, out in zip(cot_idx, outputs):
            s = samples[i]
            cont_ids = torch.tensor(out.outputs[0].token_ids, dtype=torch.long).unsqueeze(0)
            cont_mask = torch.ones_like(cont_ids)
            cache[i] = {
                "prompt_input_ids": s["prompt_ids"].unsqueeze(0),
                "prompt_attention_mask": s["prompt_mask"].unsqueeze(0),
                "chosen_input_ids": cont_ids,
                "chosen_attention_mask": cont_mask,
            }
        del llm
        torch.cuda.empty_cache()

    assert all(c is not None for c in cache)
    tmp_path = f"{cache_file}.tmp{os.getpid()}"
    torch.save(cache, tmp_path)
    os.replace(tmp_path, cache_file)
    print(f"[pregenerate_tgkd_cache] done: {len(cache)} entries cached at {cache_file}")


if __name__ == "__main__":
    main()
