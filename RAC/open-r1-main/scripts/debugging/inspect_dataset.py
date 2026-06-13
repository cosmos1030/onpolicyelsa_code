"""
데이터셋 구조 확인 스크립트
Usage:
  python scripts/inspect_dataset.py                            # OpenR1-Math-220k
  python scripts/inspect_dataset.py <path_or_dataset_name>    # 로컬 Arrow 등
"""

import sys
from datasets import load_dataset, load_from_disk
from pathlib import Path

target = sys.argv[1] if len(sys.argv) > 1 else "open-r1/OpenR1-Math-220k"

# ── 로드 ────────────────────────────────────────────────────────────────────
print(f"Loading: {target}")
try:
    ds = load_dataset(target, split="train")
except Exception:
    ds_raw = load_from_disk(target)
    ds = ds_raw["train"] if hasattr(ds_raw, "keys") else ds_raw

print(f"\n=== Overview ===")
print(f"Rows     : {len(ds):,}")
print(f"Columns  : {ds.column_names}")

# ── 컬럼별 샘플 값 ───────────────────────────────────────────────────────────
print("\n\n=== Sample Row (index 0) ===")
row = ds[0]
for col, val in row.items():
    if isinstance(val, str):
        preview = val[:400] + ("..." if len(val) > 400 else "")
        print(f"\n[{col}] (len={len(val)})\n{preview}")
    elif isinstance(val, list):
        print(f"\n[{col}] (list, len={len(val)})")
        for i, item in enumerate(val[:2]):
            s = str(item)
            print(f"  [{i}]: {s[:300]}{'...' if len(s) > 300 else ''}")
    else:
        print(f"\n[{col}]: {val}")
