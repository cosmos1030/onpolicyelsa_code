"""Migrate an already-tokenized MixedTextDataset cache to the new
min_len-filtered cache key, without re-tokenizing anything — just load the
old pickle, drop samples shorter than MIN_SAMPLE_LEN, save under the new
(min_len-inclusive) cache path.
"""
import sys
sys.path.insert(0, "/home1/doyoonkim/projects/elsa")

import os
import pickle
from lib.gkd_admm_trainer import _dataset_cache_path, _tokenizer_identity, MixedTextDataset
from transformers import AutoTokenizer

MODEL = "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH = "/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"
CACHE_DIR = "/home1/doyoonkim/projects/elsa/.cache/datasets"
OLD_CACHE = os.path.join(CACHE_DIR, "fc4dba4f8c78.pkl")  # pre-min_len-filter cache (200k samples)
MIN_SAMPLE_LEN = MixedTextDataset.MIN_SAMPLE_LEN

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
new_cache_path = _dataset_cache_path(
    CACHE_DIR, DATA_PATH, _tokenizer_identity(tokenizer),
    cls="MathCotKD", max_len=2048, max_prompt_len=512,
    nsamples=None, seed=42, min_len=MIN_SAMPLE_LEN,
)
print(f"new cache path: {new_cache_path}")

if os.path.exists(new_cache_path):
    print("already exists, nothing to do")
    sys.exit(0)

print(f"loading old cache: {OLD_CACHE}")
with open(OLD_CACHE, "rb") as f:
    samples = pickle.load(f)
print(f"loaded {len(samples)} samples")

filtered = [s for s in samples if s["input_ids"].shape[-1] >= MIN_SAMPLE_LEN]
print(f"kept {len(filtered)} samples (dropped {len(samples) - len(filtered)} shorter than {MIN_SAMPLE_LEN})")

tmp_path = f"{new_cache_path}.tmp{os.getpid()}"
with open(tmp_path, "wb") as f:
    pickle.dump(filtered, f)
os.replace(tmp_path, new_cache_path)
print(f"wrote {new_cache_path}")
