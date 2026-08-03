"""One-off cache pre-builder for MixedTextDataset (OT80/FW20), so GPU sweep
jobs find the cache already built instead of each redundantly cold-building
it (or, worse, non-rank0 ranks timing out waiting on a build that isn't
happening on their own node).
"""
import sys
sys.path.insert(0, "/home1/doyoonkim/projects/elsa")

from transformers import AutoTokenizer
from lib.gkd_admm_trainer import MixedTextDataset

MODEL = "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH = "/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
ds = MixedTextDataset(
    jsonl_path=DATA_PATH,
    tokenizer=tokenizer,
    max_len=2048,
    max_prompt_len=512,
    nsamples=None,
    seed=42,
)
print(f"cache built/loaded: {len(ds)} samples")
