"""Verify a saved checkpoint's Linear weights actually satisfy 2:4 semi-structured
sparsity: every contiguous group of 4 values along the pruning dimension has
exactly <=2 nonzeros (>=2 zeros). Reports per-tensor and aggregate violation
counts. CPU-only, no torch.distributed / no GPU needed -- just reads
safetensors shards directly.

Usage: python verify_24_structure.py <model_dir>
"""
import sys
import json
import os
from collections import OrderedDict

import torch
from safetensors import safe_open

MODEL_DIR = sys.argv[1]


def iter_linear_weights(model_dir):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_path = os.path.join(model_dir, "model.safetensors")
    by_shard = OrderedDict()
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        # group by shard file to avoid reopening repeatedly
        for name, shard in weight_map.items():
            if not name.endswith(".weight"):
                continue
            by_shard.setdefault(shard, []).append(name)
    elif os.path.exists(single_path):
        with safe_open(single_path, framework="pt", device="cpu") as f:
            names = [n for n in f.keys() if n.endswith(".weight")]
        by_shard["model.safetensors"] = names
    else:
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_dir}")

    for shard, names in by_shard.items():
        path = os.path.join(model_dir, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            for name in names:
                # embed_tokens/lm_head are never part of N:M pruning (only
                # decoder-block Linear layers are targeted) -- skip them so
                # they don't get misreported as violations.
                if "embed_tokens" in name or "lm_head" in name:
                    continue
                t = f.get_tensor(name)
                if t.dim() != 2:
                    continue
                yield name, t


def check_2_4(t, group=4, keep=2):
    """t: 2D tensor, [out_features, in_features]. 2:4 groups run along dim=1
    (in_features), matching this codebase's N:M pruning convention (grouping
    along the reduction/input dimension)."""
    out_f, in_f = t.shape
    if in_f % group != 0:
        return None  # not evenly divisible, skip
    tg = t.reshape(out_f, in_f // group, group)
    nz = (tg != 0).sum(dim=-1)
    violations = (nz > keep).sum().item()
    total_groups = out_f * (in_f // group)
    return violations, total_groups


def main():
    total_violations = 0
    total_groups = 0
    n_checked = 0
    n_skipped = 0
    worst = []
    for name, t in iter_linear_weights(MODEL_DIR):
        res = check_2_4(t)
        if res is None:
            n_skipped += 1
            continue
        v, g = res
        n_checked += 1
        total_violations += v
        total_groups += g
        if v > 0:
            worst.append((name, v, g))

    print(f"Checked {n_checked} 2D weight tensors ({n_skipped} skipped, non-4-divisible or non-2D)")
    print(f"Total groups: {total_groups:,}")
    print(f"Total violations (>2 nonzero in a group of 4): {total_violations:,}")
    if total_groups > 0:
        print(f"Violation rate: {100.0 * total_violations / total_groups:.6f}%")
    if worst:
        print(f"\n{len(worst)} tensors with violations (worst 10):")
        for name, v, g in sorted(worst, key=lambda x: -x[1])[:10]:
            print(f"  {name}: {v}/{g} groups violated ({100.0*v/g:.4f}%)")
    else:
        print("\nNo violations found -- checkpoint is fully 2:4-structured.")


if __name__ == "__main__":
    main()
