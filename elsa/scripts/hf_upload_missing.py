#!/usr/bin/env python3
"""Upload missing models to HuggingFace."""
import os, sys, time
from huggingface_hub import HfApi, create_repo, upload_folder

HF_TOKEN = open(os.path.expanduser("~/.hf_token")).read().strip()
api = HfApi(token=HF_TOKEN)
MODELS_DIR = "/home1/doyoonkim/projects/elsa/models"

UPLOADS = [
    # (local_subdir, hf_repo_id, wbid_for_reference)
    # ── GMP NTP+KD+OPKD 4B ──────────────────────────────────────
    ("gmp_s50pct_lr0.0001_onpol_lmda0.33_20260717_010613",
     "cosmos1030/qwen3-4b-gmp-ntp-kd-opkd-s50pct", "40t0ae56"),
    ("gmp_s60pct_lr0.0001_onpol_lmda0.33_20260717_011710",
     "cosmos1030/qwen3-4b-gmp-ntp-kd-opkd-s60pct", "jh8zxskl"),
    ("gmp_s70pct_lr0.0001_onpol_lmda0.33_20260717_005837",
     "cosmos1030/qwen3-4b-gmp-ntp-kd-opkd-s70pct", "5spei7jy"),
    # ── GMP lasso-only 4B ───────────────────────────────────────
    ("gmp_s50pct_lr0.0001_onpol_lmda0.33_20260718_021459",
     "cosmos1030/qwen3-4b-gmp-lasso-s50pct", "806ju8bs"),
    ("gmp_s60pct_lr0.0001_onpol_lmda0.33_20260718_030756",
     "cosmos1030/qwen3-4b-gmp-lasso-s60pct", "38345k6x"),
    ("gmp_s70pct_lr0.0001_onpol_lmda0.33_20260718_031304",
     "cosmos1030/qwen3-4b-gmp-lasso-s70pct", "3pd1uzuu"),
    # ── TR-GMP dwup checkpoints 4B ──────────────────────────────
    ("gmp_s70pct_lr0.0001_onpol_lmda0.33_sp50_20260723_015417",
     "cosmos1030/qwen3-4b-gmp-tr-dwup-s50pct", "zzgsbclt"),
    ("gmp_s70pct_lr0.0001_onpol_lmda0.33_sp50_20260722_192639",
     "cosmos1030/qwen3-4b-gmp-tr-dwup-lasso-s50pct", "dnuosqxr"),
    ("gmp_s70pct_lr0.0001_onpol_lmda0.33_sp60_20260722_210148",
     "cosmos1030/qwen3-4b-gmp-tr-dwup-lasso-s60pct-v1", "t4q9ygw2"),
    ("gmp_s70pct_lr0.0001_onpol_lmda0.33_sp60_20260723_100603",
     "cosmos1030/qwen3-4b-gmp-tr-dwup-lasso-s60pct-v2", "8eofnv1c"),
    # ── ELSA NTP-ADMM 4B ────────────────────────────────────────
    ("1cfa9a7208912126459214e8b04321603b3df60c_pruned0.5_kd_unknown_admm_lr5e-05_lmda0.001_kdlam1.0_20260721_105101",
     "cosmos1030/elsa-ntp-admm-qwen3-4b-s50pct", "ng28it84"),
    ("1cfa9a7208912126459214e8b04321603b3df60c_pruned0.6_kd_unknown_admm_lr5e-05_lmda0.005_kdlam1.0_20260722_171924",
     "cosmos1030/elsa-ntp-admm-qwen3-4b-s60pct", "fqcfkllx"),
    ("1cfa9a7208912126459214e8b04321603b3df60c_pruned0.7_kd_unknown_admm_lr0.0001_lmda0.005_kdlam1.0_20260722_192002",
     "cosmos1030/elsa-ntp-admm-qwen3-4b-s70pct", "0uqv2ogw"),
    # ── ALPS 4B ─────────────────────────────────────────────────
    ("qwen3_4b_alps_s50pct", "cosmos1030/alps-qwen3-4b-s50pct", "54q7yvlw"),
    ("qwen3_4b_alps_s60pct", "cosmos1030/alps-qwen3-4b-s60pct", "jn0vsqi1"),
    ("qwen3_4b_alps_s70pct", "cosmos1030/alps-qwen3-4b-s70pct", "alps4b_s70_local"),
    # ── SparseLLM 4B ────────────────────────────────────────────
    ("qwen3_4b_sparsellm_s50pct", "cosmos1030/sparsellm-qwen3-4b-s50pct", "lurwi0ni"),
    ("qwen3_4b_sparsellm_s60pct", "cosmos1030/sparsellm-qwen3-4b-s60pct", "kh6indij"),
    ("qwen3_4b_sparsellm_s70pct", "cosmos1030/sparsellm-qwen3-4b-s70pct", "50ugpxrr"),
    # ── GMP NTP+KD 8B ───────────────────────────────────────────
    ("gmp_s50pct_lr0.0001_20260722_042556",
     "cosmos1030/qwen3-8b-gmp-ntp-kd-s50pct", "n21csufv"),
    # ── SAFE 1.7B ───────────────────────────────────────────────
    ("70d244cc86ccca08cf5af4e1e306ecf908b1ad5e_safe_s50pct_20260722_1640",
     "cosmos1030/safe-qwen3-1.7b-s50pct", "vs2qi1zh"),
    ("70d244cc86ccca08cf5af4e1e306ecf908b1ad5e_safe_s60pct_20260722_1646",
     "cosmos1030/safe-qwen3-1.7b-s60pct", "ig0ea9ms"),
    ("70d244cc86ccca08cf5af4e1e306ecf908b1ad5e_safe_s70pct_20260722_1641",
     "cosmos1030/safe-qwen3-1.7b-s70pct", "ojb131fn"),
]

LOG = "/home1/doyoonkim/hf_upload_logs/hf_upload_log.txt"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

log(f"Starting upload of {len(UPLOADS)} models")

results = {}  # wbid → hf_repo

for subdir, repo_id, wbid in UPLOADS:
    local_path = os.path.join(MODELS_DIR, subdir)
    if not os.path.isfile(os.path.join(local_path, "config.json")):
        log(f"SKIP {repo_id} — model not found at {local_path}")
        continue

    # Check if already exists on HF
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        log(f"EXISTS {repo_id} — skipping")
        results[wbid] = repo_id
        continue
    except Exception:
        pass

    log(f"UPLOADING {repo_id} from {subdir} ...")
    try:
        create_repo(repo_id=repo_id, repo_type="model", token=HF_TOKEN, exist_ok=True)
        upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type="model",
            token=HF_TOKEN,
            commit_message=f"Upload {subdir}",
        )
        log(f"  DONE {repo_id}")
        results[wbid] = repo_id
    except Exception as e:
        log(f"  ERROR {repo_id}: {e}")

log("Upload complete.")
log("Results:")
for wbid, repo in results.items():
    log(f"  {wbid} → {repo}")

# Write results to a file for artifact update
import json
with open("/home1/doyoonkim/hf_upload_logs/hf_upload_results.json", "w") as f:
    json.dump(results, f, indent=2)
log("Saved results to hf_upload_results.json")
