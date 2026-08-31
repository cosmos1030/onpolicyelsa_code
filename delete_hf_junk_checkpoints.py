#!/usr/bin/env python3
"""Delete old/junk HF checkpoint repos under cosmos1030.

Two categories:
1. EXTRA_REPOS: 4 named repos, each a sweep sibling explicitly self-described
   as "collapsed"/"거의 붕괴" in the reasoning_bench_8192 dashboard, and NOT
   the wbid actually cited for its sparsity/method in the current results
   table (a different sweep point from the same family is the one kept).
2. Everything created before CUTOFF (2026-07-01): confirmed zero overlap
   with the "nostrip8192"/OT80/FW20 naming convention that every checkpoint
   currently cited in the dashboard uses -- these predate that data-fix
   entirely (either unrelated pre-project repos from 2025, or pre-methodology-
   fix pruning sweeps from May-June 2026).

Run with: /home1/doyoonkim/miniconda3/envs/rac/bin/python3 delete_hf_junk_checkpoints.py
"""
from huggingface_hub import HfApi
import os
import datetime

api = HfApi(token=os.environ.get('HF_TOKEN') or open(os.path.expanduser('~/.hf_token')).read().strip())

CUTOFF = datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc)

EXTRA_REPOS = [
    "cosmos1030/elsa-ntp-cot-s60pct-lr5e-5-lmda5e-4_20260814_075601",   # wbid pc7whd4w, 1.7B ELSA S60 sweep, "거의 붕괴"
    "cosmos1030/elsa-ntp-cot-s70pct-lr1e-4-lmda2e-3_20260814_091227",   # wbid zp9u3lwf, 1.7B ELSA S70 sweep, "붕괴"
    "cosmos1030/elsa-ntp-cot-s70pct-lr1e-4-lmda1e-2_20260814_094347",   # wbid eo6qezka, 1.7B ELSA S70 sweep, "붕괴"
    "cosmos1030/alps-s70pct_20260811_015229",                          # wbid 7mnmil85, 1.7B ALPS S70, "collapsed at s70"
]

print("Listing all cosmos1030 model repos...")
models = list(api.list_models(author='cosmos1030', full=True))
old = sorted([m.id for m in models if m.created_at < CUTOFF], key=str)

repos = sorted(set(old) | set(EXTRA_REPOS))
print(f"\n{len(old)} repos created before {CUTOFF.date()}, + {len(EXTRA_REPOS)} named extras "
      f"= {len(repos)} total to delete.\n")
for r in repos:
    print(f"  - {r}")

resp = input(f"\nType 'yes' to permanently delete all {len(repos)} repos above: ").strip()
if resp != "yes":
    print("Aborted.")
    raise SystemExit(0)

ok, fail = 0, 0
for r in repos:
    try:
        api.delete_repo(repo_id=r, repo_type="model")
        print(f"DELETED: {r}")
        ok += 1
    except Exception as e:
        print(f"FAILED: {r} -> {e}")
        fail += 1

print(f"\nDone: {ok} deleted, {fail} failed.")
