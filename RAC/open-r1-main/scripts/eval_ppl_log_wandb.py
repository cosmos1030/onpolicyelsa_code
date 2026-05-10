"""
Evaluate c4/wikitext2 PPL + MATH-500 for saved pruned models and log to existing wandb runs.
Usage:
    python scripts/eval_ppl_log_wandb.py
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

ENTITY  = "dyk6208-gwangju-institute-of-science-and-technology"
PROJECT = "gmp_qwen3_1.5b"
MODEL_BASE = "/home1/doyoonkim/projects/RAC/open-r1-main/models"
SEQLEN  = 2048
MATH500_MAX_NEW_TOKENS = 8192
MATH500_MAX_SAMPLES    = None  # None = all 500

# run_id -> model_dir (relative to MODEL_BASE)
RUNS = {
    # CoT calibration sweep (zcovudr7)
    "zg7q7q5s": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_pruned_30_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k_.jsonl",
    "4cd4w3d5": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_pruned_50_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k_.jsonl",
    "vt7rwx6k": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_pruned_70_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k_.jsonl",
    "74osakoo": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_pruned_90_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k_.jsonl",
    # Prompt-only calibration sweep (syz1lwr5)
    "vn21htne": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_pruned_30_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__OpenR1-Math-220k",
    "z3y076ih": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_pruned_70_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3__OpenR1-Math-220k",
}


def eval_ppl(model_path):
    print(f"  [PPL] Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.seqlen = SEQLEN
    model.eval()

    results = {}
    for dataset in ["wikitext2", "c4"]:
        print(f"  [PPL] Evaluating {dataset}...", flush=True)
        _, testloader = get_loaders(dataset, seed=0, seqlen=SEQLEN, tokenizer=tokenizer)
        with torch.no_grad():
            ppl = calculate_ppl(model, testloader, tokenizer, bs=1)
        results[f"ppl_test({dataset})"] = ppl
        print(f"  [PPL] {dataset} = {ppl:.2f}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    for run_id, model_subdir in RUNS.items():
        model_path = os.path.join(MODEL_BASE, model_subdir)
        if not os.path.isdir(model_path):
            print(f"SKIP {run_id}: model not found at {model_path}", flush=True)
            continue

        print(f"\n=== Run {run_id} | {model_subdir[:60]}... ===", flush=True)

        # Init wandb run first so lighteval_math500 logs to it
        run = wandb.init(
            entity=ENTITY,
            project=PROJECT,
            id=run_id,
            resume="allow",
        )

        # --- PPL ---
        ppl_results = eval_ppl(model_path)
        wandb.log(ppl_results)
        wandb.run.summary.update(ppl_results)
        print(f"  PPL logged: {ppl_results}", flush=True)

        # --- MATH-500 ---
        print(f"  [MATH500] Running lighteval...", flush=True)
        output_dir = os.path.join(model_path, "lighteval_math500")
        try:
            pass_at_1 = run_lighteval_math500(
                model_path=model_path,
                output_dir=output_dir,
                max_new_tokens=MATH500_MAX_NEW_TOKENS,
                max_samples=MATH500_MAX_SAMPLES,
                log_to_wandb=True,
            )
            wandb.run.summary.update({"math500_pass@1": pass_at_1})
            print(f"  MATH500 pass@1 = {pass_at_1:.4f}", flush=True)
        except Exception as e:
            print(f"  [MATH500] ERROR: {e}", flush=True)

        wandb.finish()


if __name__ == "__main__":
    main()
