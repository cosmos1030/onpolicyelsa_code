"""
Verify FisherAccumulator (Adam exp_avg_sq) behaves correctly:
1. exp_avg_sq is populated after optimizer steps
2. importance scores are non-zero / non-degenerate
3. sparsity actually increases at mask updates
4. Compare importance distribution vs old hand-rolled EMA
"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.gmp_trainer import FisherAccumulator, GradualMaskManager, _cubic_sparsity

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# tiny model proxy
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64, bias=False) for _ in range(4)])
    def forward(self, x):
        for l in self.layers: x = torch.relu(l(x))
        return x.mean()

model = TinyModel().to(device)
named_params = {f"model.layers.{i}.weight": model.layers[i].weight for i in range(4)}

# ── new: Adam exp_avg_sq Fisher ────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)
fisher_new = FisherAccumulator(named_params, optimizer)

# ── old: hand-rolled EMA Fisher (for comparison) ───────────────────────────
beta = 0.999
F_old = {n: torch.zeros_like(p.data, dtype=torch.float32) for n, p in named_params.items()}
old_step = 0

x = torch.randn(8, 64, device=device)

print(f"\n{'step':>5}  {'imp_new_mean':>14}  {'imp_old_mean':>14}  {'ratio':>8}  {'sparsity':>10}")
print("-" * 65)

maskmgr = GradualMaskManager(named_params)
total_steps = 40
mask_interval = 10
final_sparsity = 0.5
warmup_steps = 5

optimizer.zero_grad()
for step in range(1, total_steps + 1):
    loss = model(x)
    loss.backward()

    # old EMA update (before optimizer step, same as before)
    old_step += 1
    for n, p in named_params.items():
        if p.grad is not None:
            g = p.grad.data.float()
            F_old[n].mul_(beta).addcmul_(g, g, value=1 - beta)

    fisher_new.update()  # no-op
    optimizer.step()
    optimizer.zero_grad()
    maskmgr.apply()

    if step % mask_interval == 0:
        target_sp = _cubic_sparsity(step, total_steps, final_sparsity, warmup_steps)

        # compute importance stats for both
        imp_new_vals, imp_old_vals = [], []
        for n, p in named_params.items():
            imp_n = fisher_new.importance(n, p)
            imp_new_vals.append(imp_n.float().flatten())

            f_o = F_old[n] / (1.0 - beta ** old_step)
            imp_o = f_o * p.data.float() ** 2
            if imp_o.sum() == 0: imp_o = p.data.float() ** 2
            imp_old_vals.append(imp_o.flatten())

        imp_new = torch.cat(imp_new_vals)
        imp_old = torch.cat(imp_old_vals)

        ratio = (imp_new.mean() / imp_old.mean().clamp(min=1e-12)).item()

        maskmgr.update(fisher_new, target_sp)
        real_sp = maskmgr.current_sparsity()

        print(f"{step:>5}  {imp_new.mean().item():>14.6f}  {imp_old.mean().item():>14.6f}  {ratio:>8.4f}  {real_sp:>10.4f}")

        # sanity checks
        assert imp_new.sum() > 0, "FAIL: new importance all-zero"
        assert imp_old.sum() > 0, "FAIL: old importance all-zero"
        assert abs(ratio - 1.0) < 0.2, f"WARN: ratio={ratio:.4f} deviates >20% from old"
        assert abs(real_sp - target_sp) < 0.05, f"FAIL: sparsity {real_sp:.4f} vs target {target_sp:.4f}"

print("\nAll checks passed.")
