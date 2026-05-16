"""Quick test: save a tiny model and push to HF Hub with new repo ID format."""
import os
import tempfile
from datetime import datetime
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

def fmt_float(v):
    s = f"{v:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e")

now = datetime.now().strftime("%Y%m%d_%H%M%S")
lr = 1e-4
sparsity = 0.5
hub_repo = f"cosmos1030/gmp-s{int(sparsity*100)}pct-lr{fmt_float(lr)}_{now}"
print(f"repo_id: {hub_repo}  (len={len(hub_repo)})")

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)

with tempfile.TemporaryDirectory() as tmpdir:
    tok.save_pretrained(tmpdir)
    # write a minimal config so it's a valid repo
    import shutil, json
    cfg_src = os.path.join(MODEL_PATH, "config.json")
    shutil.copy(cfg_src, tmpdir)

    print("Creating HF repo...")
    api = HfApi()
    api.create_repo(repo_id=hub_repo, exist_ok=True, private=True)

    print("Uploading...")
    api.upload_folder(folder_path=tmpdir, repo_id=hub_repo, commit_message="debug push_to_hub test")
    print(f"SUCCESS: https://huggingface.co/{hub_repo}")

    print("Deleting test repo...")
    api.delete_repo(repo_id=hub_repo)
    print("Deleted.")
