"""Pre-generate the teacher-gen-KD (TGKD) chosen-continuation cache once, offline,
via vLLM, so real TR-GMP training jobs (gmp_teacher_gen_kd=true) load an
already-built cache instead of generating 16k+ continuations inline at
startup. Mirrors exactly how lib/gmp_trainer.py builds train_dataset /
NTPPromptWrapper / the generate_chosen_cache() call, so the cache key
(prompt_path string + n_pairs + max_new_tokens + temperature) matches what a
real run with the same flags will look up.
"""
import argparse
import torch
from transformers import AutoTokenizer

from lib.gkd_admm_trainer import MixedTextDataset, NTPPromptWrapper
from lib.gmp_dpo import generate_chosen_cache
from lib.utils import get_llm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data_path", required=True,
                   help="Training data path (disjoint from --opd_prompt_path).")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max_prompt_len", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--cache_dir", default="/home1/doyoonkim/projects/elsa/.cache/dpo_chosen")
    p.add_argument("--gpu_mem", type=float, default=0.85)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # generate_chosen_cache's vLLM path unconditionally calls dense_model.cpu()
    # (offloads it before vLLM claims GPU memory) and dense_model.to(device)
    # afterward -- load the real dense model so that works, even though vLLM
    # (not this in-process model) does the actual generation.
    dense_model = get_llm(args.model, args.seqlen)

    train_dataset = MixedTextDataset(
        jsonl_path=args.data_path,
        tokenizer=tokenizer,
        max_prompt_len=args.max_prompt_len,
        max_len=args.seqlen,
        append_eos=False,
        nsamples=None,
    )
    prompt_dataset = NTPPromptWrapper(train_dataset)

    n_pairs = args.steps * args.batch_size * args.grad_accum
    gbs = args.batch_size * args.grad_accum
    prompt_path_key = f"{args.data_path}|ntp_prompt_wrapper|gbs={gbs}"

    print(f"[pregenerate_tgkd_cache] n_pairs={n_pairs} prompt_path_key={prompt_path_key!r} "
          f"dataset_size={len(prompt_dataset)}")

    cache = generate_chosen_cache(
        dense_model=dense_model,
        tokenizer=tokenizer,
        prompt_dataset=prompt_dataset,
        n_pairs=n_pairs,
        gen_batch_size=64,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device="cuda",
        cache_dir=args.cache_dir,
        prompt_path=prompt_path_key,
        store_teacher_logps=False,
        use_vllm=True,
        model_path=args.model,
    )
    print(f"[pregenerate_tgkd_cache] done: {len(cache)} entries cached under {args.cache_dir}")


if __name__ == "__main__":
    main()
