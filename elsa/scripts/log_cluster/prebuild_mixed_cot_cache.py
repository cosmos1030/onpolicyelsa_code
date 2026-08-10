"""One-off cache pre-builder for MixedTextDataset (log_cluster paths), so GPU
sweep jobs find the cache already built instead of each redundantly
cold-building it on a GPU node.
"""
import sys
sys.path.insert(0, "/home/doyoonkim/projects/onpolicyelsa_code/elsa")

from transformers import AutoTokenizer
from lib.gkd_admm_trainer import MixedTextDataset

MODEL = "Qwen/Qwen3-1.7B"
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "/home/doyoonkim/projects/onpolicyelsa_code/elsa/data/ot3_fineweb_200k_qwen3_thinkstrip.jsonl"
MAX_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
ds = MixedTextDataset(
    jsonl_path=DATA_PATH,
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    max_prompt_len=512,
    nsamples=None,
    seed=42,
)
print(f"cache built/loaded: {len(ds)} samples")
