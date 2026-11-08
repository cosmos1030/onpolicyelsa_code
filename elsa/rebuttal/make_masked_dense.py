"""Write an HF model dir = original dense weights * mask, so --gmp_fixed_mask
picks the mask up from its zero pattern.

mask source: nonzero pattern of --mask_dir (projection weights only).
--swap_from_final: with --mask_dir=reach and this=final dir, build the random-swap
control (prune N random reach-kept, revive M random reach-pruned; N/M = the
real reach->final counts).
"""
import argparse, glob, shutil, os
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def load_masks(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k.endswith('.weight') and any(t in k for t in PROJ):
                    out[k] = (s.get_tensor(k) != 0)
    return out


ap = argparse.ArgumentParser()
ap.add_argument('--dense', required=True)
ap.add_argument('--mask_dir', required=True)
ap.add_argument('--swap_from_final', default='')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--out', required=True)
a = ap.parse_args()

m = load_masks(a.mask_dir)
keys = sorted(m)
if a.swap_from_final:
    mf = load_masks(a.swap_from_final)
    n_rm = sum((m[k] & ~mf[k]).sum().item() for k in keys)
    n_add = sum((~m[k] & mf[k]).sum().item() for k in keys)
    flat = torch.cat([m[k].flatten() for k in keys])
    g = torch.Generator().manual_seed(a.seed)
    kept = flat.nonzero().squeeze(1); pr = (~flat).nonzero().squeeze(1)
    f2 = flat.clone()
    f2[kept[torch.randperm(kept.numel(), generator=g)[:n_rm]]] = False
    f2[pr[torch.randperm(pr.numel(), generator=g)[:n_add]]] = True
    o = 0
    for k in keys:
        n = m[k].numel(); m[k] = f2[o:o + n].view_as(m[k]); o += n
    print(f"random swap seed={a.seed}: pruned {n_rm:,} revived {n_add:,}")

model = AutoModelForCausalLM.from_pretrained(a.dense, torch_dtype=torch.bfloat16)
sd = dict(model.named_parameters())
tot = kept = 0
with torch.no_grad():
    for k in keys:
        sd[k].mul_(m[k].to(sd[k].dtype))
        tot += m[k].numel(); kept += m[k].sum().item()
print(f"prunable={tot:,} kept={kept:,} sparsity={1 - kept / tot:.5f}")
model.save_pretrained(a.out, safe_serialization=True)
AutoTokenizer.from_pretrained(a.dense).save_pretrained(a.out)
print("saved", a.out)
