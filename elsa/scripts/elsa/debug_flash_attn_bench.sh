#!/bin/bash
#SBATCH --job-name=flash_bench
#SBATCH --partition=A5000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j_flash_bench.out
#SBATCH -t 0-00:10:00

set -euo pipefail
echo "Node: $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<'EOF'
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

prompt = "Solve step by step: What is the sum of all prime numbers less than 100?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
MAX_NEW_TOKENS = 512
N_RUNS = 3

def benchmark(attn_impl):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    # warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=32, do_sample=False)

    times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        n_tokens = out.shape[1] - inputs["input_ids"].shape[1]
        times.append((elapsed, n_tokens))

    avg_time = sum(t for t, _ in times) / N_RUNS
    avg_tokens = sum(n for _, n in times) / N_RUNS
    print(f"[{attn_impl:20s}] avg {avg_time:.2f}s | {avg_tokens/avg_time:.1f} tok/s ({int(avg_tokens)} tokens)")

    del model
    torch.cuda.empty_cache()

benchmark("eager")
benchmark("flash_attention_2")
EOF
echo "=== Done ==="
date
