#!/usr/bin/env python3
"""Upload missing HF models and print wbid→repo mapping for artifact update."""
import os, sys, json

HF_TOKEN = open(os.path.expanduser("~/.hf_token")).read().strip()
HF_USER = "cosmos1030"

# Models to upload from disk: (wbid, local_path, hf_repo_name)
UPLOADS = [
    # Qwen3-4B SparseGPT
    ("h172fegj", "/home1/doyoonkim/projects/elsa/models/qwen3_4b_sgpt_s50pct_n128",
     f"{HF_USER}/qwen3-4b-sgpt-s50pct-ot80fw20"),
    ("d9oyfmwh", "/home1/doyoonkim/projects/elsa/models/qwen3_4b_sgpt_s60pct_n128",
     f"{HF_USER}/qwen3-4b-sgpt-s60pct-ot80fw20"),
    # S70 SparseGPT also exists on disk — upload it too
    ("s70_sgpt", "/home1/doyoonkim/projects/elsa/models/qwen3_4b_sgpt_s70pct_n128",
     f"{HF_USER}/qwen3-4b-sgpt-s70pct-ot80fw20"),
    # Qwen3-4B GMP TR-GMP kl=0.01/0.02 S50
    ("jjajjn4p", "/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_sp50_20260715_235752",
     f"{HF_USER}/qwen3-4b-gmp-tr-ntpkd-s50pct-kl001"),
    ("xcthflzv", "/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_sp50_20260715_232640",
     f"{HF_USER}/qwen3-4b-gmp-tr-ntpkd-s50pct-kl002"),
    # Qwen3-4B GMP TR-GMP kl=0.01/0.02 S60
    ("rp4u5w1g", "/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_sp60_20260716_002628",
     f"{HF_USER}/qwen3-4b-gmp-tr-ntpkd-s60pct-kl001"),
    ("buj4a1tn", "/home1/doyoonkim/projects/elsa/models/gmp_s70pct_lr0.0001_sp60_20260715_234524",
     f"{HF_USER}/qwen3-4b-gmp-tr-ntpkd-s60pct-kl002"),
]

# Models already on HF — just print for artifact update
ALREADY_UPLOADED = {
    "jmagj6yb": "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260712_082548",
    "qka9u28l": "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260712_062009",
    "f67yydns":  "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260712_100353",
    "txfgrpg5": "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260712_084045",
    "crt5l9ty": "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260712_124610",
    "i8n23nca": "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260712_082547",
    "oomrjlla": "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260712_101701",
}

from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)

results = dict(ALREADY_UPLOADED)

for wbid, local_path, repo_id in UPLOADS:
    if not os.path.isfile(os.path.join(local_path, "config.json")):
        print(f"[SKIP] {wbid}: no config.json in {local_path}", flush=True)
        continue
    print(f"[UPLOAD] {wbid} → {repo_id} from {local_path}", flush=True)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload pruned model ({wbid})",
        )
        print(f"[DONE]  {wbid} → https://huggingface.co/{repo_id}", flush=True)
        results[wbid] = repo_id
    except Exception as e:
        print(f"[ERROR] {wbid}: {e}", flush=True)

print("\n=== FINAL MAPPING (wbid → hf_repo) ===")
for wbid, repo in sorted(results.items()):
    print(f"{wbid}: {repo}")

# Save to JSON for artifact update script
out = "/home1/doyoonkim/projects/elsa/scripts/hf_upload_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out}")
