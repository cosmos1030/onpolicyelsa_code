import sys
sys.path.insert(0, "/home1/doyoonkim/projects/elsa")
from transformers import AutoModelForCausalLM
import torch

model_path = "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)

from lib.gmp_trainer import _find_linear_weights
named = _find_linear_weights(model)
total = sum(p.numel() for p in named.values())
print("num linear-weight tensors:", len(named))
print("total prunable weights:", total, f"({total/1e9:.3f}B)")

flat = torch.cat([p.detach().flatten().float()**2 for p in named.values()])
print("total elements in flat:", flat.numel())

qs = [0.10, 0.30, 0.50, 0.70, 0.90]
vals = torch.quantile(flat, torch.tensor(qs), interpolation='linear')
for q, v in zip(qs, vals.tolist()):
    print(f"  q={q:.2f} -> weight^2 threshold = {v:.3e}")

for q, v in zip(qs, vals.tolist()):
    lo, hi = v*0.99, v*1.01
    cnt = ((flat >= lo) & (flat <= hi)).sum().item()
    print(f"  q={q:.2f} thr={v:.3e}  count within +/-1% band = {cnt}")
