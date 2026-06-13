"""
2:4 semi-structured sparsity unit test.

Checks:
1. _nm_mask: 최대 50% sparsity (2:4 → 각 4개 중 2개 보호 → 최대 50% pruning)
2. 점진적 schedule: step 0 (0%) → step T (50%) 동안 단조증가
3. 각 group-of-4 내 보호된 top-2는 항상 살아있는지
4. 전체 GradualMaskManager loop (10 update steps)
"""
import sys
sys.path.insert(0, '/home1/doyoonkim/projects/elsa')

import torch
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ──────────────────────────────────────────────
# 1. _nm_mask standalone test
# ──────────────────────────────────────────────
from lib.gmp_trainer import GradualMaskManager, FisherAccumulator

device = torch.device('cpu')

# Small fake weight: 8 rows × 8 cols → 16 groups-of-4
torch.manual_seed(0)
W = torch.randn(8, 8)
named_params = {'fake.weight': torch.nn.Parameter(W.clone())}
mgr = GradualMaskManager(named_params, prune_n=2, prune_m=4)

# Importance = |W|
imp = W.abs()
current_pruned = torch.zeros_like(W, dtype=torch.bool)  # nothing pruned yet

print("\n=== Test 1: _nm_mask at 50% sparsity ===")
mask = mgr._nm_mask(imp, current_pruned, sparsity=0.5)  # mask=True → KEEP
keep_ratio = mask.float().mean().item()
sparsity   = 1.0 - keep_ratio
print(f"  keep ratio: {keep_ratio:.3f}, sparsity: {sparsity:.3f}")
assert abs(sparsity - 0.5) < 1e-3, f"Expected 0.5, got {sparsity}"
print("  PASS: exact 50% sparsity")

# Each group-of-4: at least 2 survive
mask_4d = mask.reshape(-1, 4)
min_keep = mask_4d.sum(dim=1).min().item()
print(f"  min kept per group-of-4: {min_keep}")
assert min_keep >= 2, f"Expected >=2 per group, got {min_keep}"
print("  PASS: each group keeps ≥2 weights")

# Top-2 per group are protected
imp_4d = imp[:, :8].reshape(-1, 4)
for g in range(imp_4d.shape[0]):
    top2 = torch.topk(imp_4d[g], 2).indices
    for idx in top2:
        assert mask_4d[g, idx].item(), f"Group {g}: top-2 index {idx} was pruned!"
print("  PASS: top-2 per group always protected")

# ──────────────────────────────────────────────
# 2. Gradual sparsity schedule: 0% → 50%
# ──────────────────────────────────────────────
print("\n=== Test 2: gradual schedule 0→50% ===")
from lib.gmp_trainer import _cubic_sparsity

steps = [0, 100, 200, 300, 400, 500]
total = 500
prev = -1.0
for s in steps:
    sp = _cubic_sparsity(s, total, 0.5)
    print(f"  step {s:3d}: sparsity={sp:.4f}")
    assert sp >= prev - 1e-6, f"Not monotone at step {s}"
    prev = sp
print("  PASS: monotone increase")

# ──────────────────────────────────────────────
# 3. GradualMaskManager.update() loop
# ──────────────────────────────────────────────
print("\n=== Test 3: GradualMaskManager full update loop ===")
torch.manual_seed(42)
W2 = torch.randn(16, 16)
p2 = torch.nn.Parameter(W2.clone())
named2 = {'layer.weight': p2}
mgr2 = GradualMaskManager(named2, prune_n=2, prune_m=4)

# Fake FisherAccumulator: importance = |W|
class FakeFisher:
    def importance(self, name, param):
        return param.data.abs()

fisher = FakeFisher()

prev_sp = 0.0
for step in range(10):
    target_sp = _cubic_sparsity(step * 50, 500, 0.5)
    mgr2.update(fisher, target_sp)
    real_sp = 1.0 - mgr2.masks['layer.weight'].float().mean().item()
    print(f"  step {step}: target={target_sp:.4f}, real={real_sp:.4f}")
    assert real_sp >= prev_sp - 1e-4, f"Sparsity decreased at step {step}"
    prev_sp = real_sp

# Final: at 50%, each group-of-4 has exactly 2 zeros
final_mask = mgr2.masks['layer.weight']
gm = final_mask.reshape(-1, 4)
zeros_per_group = (gm == 0).sum(dim=1)
print(f"  zeros per group: min={zeros_per_group.min()}, max={zeros_per_group.max()}")
assert zeros_per_group.max() <= 2, "Some group has >2 zeros (violates 2:4 constraint)"
print("  PASS: N:M constraint satisfied at final step")

# ──────────────────────────────────────────────
# 4. Weight values actually zeroed
# ──────────────────────────────────────────────
print("\n=== Test 4: zeroed weights stay zero after apply ===")
zero_positions = (~final_mask)
assert (p2.data[zero_positions] == 0).all(), "Pruned weights not zeroed in param"
print("  PASS: pruned positions are 0 in param.data")

print("\n=== ALL TESTS PASSED ===")
