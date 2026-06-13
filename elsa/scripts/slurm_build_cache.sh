#!/bin/bash
#SBATCH --job-name=build_cache
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=0-01:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/build_cache_%j.out

exec 2>&1

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

echo "Node: $(hostname)"
echo "Building dataset caches to /home1/doyoonkim/projects/elsa/.cache/datasets/"

/home1/doyoonkim/miniconda3/envs/rac/bin/python -u - << 'PYEOF'
import sys, os, time
sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
from transformers import AutoTokenizer
from lib.gkd_admm_trainer import MathCotKDDataset, MathPromptWithAnswerDataset

MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
CACHE_DIR  = "/home1/doyoonkim/projects/elsa/.cache/datasets"
os.makedirs(CACHE_DIR, exist_ok=True)

print("Loading tokenizer...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# Build MathCotKDDataset cache
print("\n[1] Building MathCotKDDataset cache...", flush=True)
t0 = time.time()
ds1 = MathCotKDDataset(
    jsonl_path="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl",
    tokenizer=tok,
    max_len=2048,
    max_prompt_len=512,
    nsamples=None,
    seed=42,
    cache_dir=CACHE_DIR,
)
print(f"  Done: {len(ds1)} samples in {time.time()-t0:.1f}s", flush=True)

# Build MathPromptWithAnswerDataset cache
print("\n[2] Building MathPromptWithAnswerDataset cache...", flush=True)
t0 = time.time()
ds2 = MathPromptWithAnswerDataset(
    jsonl_path="/home1/doyoonkim/projects/elsa/data/math_220k_with_answers.jsonl",
    tokenizer=tok,
    max_prompt_len=512,
    nsamples=None,
    seed=42,
    cache_dir=CACHE_DIR,
)
print(f"  Done: {len(ds2)} samples in {time.time()-t0:.1f}s", flush=True)

# Report cache sizes
import glob
for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*.pkl"))):
    sz = os.path.getsize(f) / 1024**3
    print(f"  {os.path.basename(f)}: {sz:.2f} GB")

print("\nAll caches built successfully!", flush=True)
PYEOF

echo "##### END #####"
