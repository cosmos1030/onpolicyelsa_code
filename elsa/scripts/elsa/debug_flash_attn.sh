#!/bin/bash
#SBATCH --job-name=flash_attn_test
#SBATCH --partition=A5000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/home1/doyoonkim/projects/elsa/output_qwen/%j_flash_attn_test.out
#SBATCH -t 0-00:10:00

set -euo pipefail
echo "Node: $(hostname)"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"

print("Loading model with flash_attention_2...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

print(f"attn_implementation: {model.config._attn_implementation}")

# Quick forward pass
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inputs)
print(f"Forward pass OK, logits shape: {out.logits.shape}")

# Quick generation
gen = model.generate(**inputs, max_new_tokens=20, do_sample=False)
print(f"Generation OK: {tokenizer.decode(gen[0], skip_special_tokens=True)}")

print("flash_attention_2 test PASSED")
EOF
echo "=== Done ==="
date
