"""
Evaluate dense (unpruned) DeepSeek-R1-Distill-Qwen-1.5B:
  - wikitext2 / c4 PPL
  - MATH-500 pass@1 + token/truncation stats
Logs to a new wandb run in gmp_qwen3_1.5b for comparison with pruned runs.

Usage:
    python scripts/eval_dense_log_wandb.py
"""
import sys
sys.path.insert(0, "/home1/doyoonkim/projects/elsa")

import gc
import os
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.eval import calculate_ppl
from lib.data import get_loaders
from lib.lighteval_math500 import run_lighteval_math500

ENTITY    = "dyk6208-gwangju-institute-of-science-and-technology"
PROJECT   = "gmp_qwen3_1.5b"
RUN_NAME  = "dense_deepseek_1.5b"
MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
SEQLEN    = 2048
MATH500_MAX_NEW_TOKENS = 8192
MATH500_MAX_SAMPLES    = 100


def main():
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=RUN_NAME,
    )

    # --- PPL ---
    print("[PPL] Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda")
    model.seqlen = SEQLEN
    model.eval()

    ppl_results = {}
    for dataset in ["wikitext2", "c4"]:
        print(f"[PPL] Evaluating {dataset}...", flush=True)
        _, testloader = get_loaders(dataset, seed=0, seqlen=SEQLEN, tokenizer=tokenizer)
        with torch.no_grad():
            ppl = calculate_ppl(model, testloader, tokenizer, bs=1)
        ppl_results[f"ppl_test({dataset})"] = ppl
        print(f"[PPL] {dataset} = {ppl:.2f}", flush=True)

    wandb.log(ppl_results)
    wandb.run.summary.update(ppl_results)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # --- MATH-500 ---
    print("[MATH500] Running lighteval...", flush=True)
    output_dir = os.path.join(MODEL_PATH, "lighteval_math500_dense")
    pass_at_1 = run_lighteval_math500(
        model_path=MODEL_PATH,
        output_dir=output_dir,
        max_new_tokens=MATH500_MAX_NEW_TOKENS,
        max_samples=MATH500_MAX_SAMPLES,
        log_to_wandb=True,
    )
    wandb.run.summary.update({"math500_pass@1": pass_at_1})
    print(f"[MATH500] pass@1 = {pass_at_1:.4f}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()
