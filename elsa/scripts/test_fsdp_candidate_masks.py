import os, sys, torch
import torch.distributed as dist

sys.path.insert(0, "/home1/doyoonkim/projects/elsa/lib")
from gmp_trainer import GradualMaskManager


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(0)

    rows, cols, prune_n, prune_m = 17, 20, 2, 4  # not evenly divisible
    full_imp_ref = torch.rand(rows, cols)  # same seed on all ranks -> identical

    total = rows * cols
    base = total // world_size
    sizes = [base] * world_size
    sizes[-1] += total - base * world_size
    offsets = [sum(sizes[:r]) for r in range(world_size)]

    class FakeParam:
        def __init__(self, flat):
            self.data = flat

    # local shard of a fake "weight" param -- this is what the local FSDP
    # shard would be; the FakeFisher below returns the pre-generated
    # full_imp_ref's corresponding local slice as "importance" so we know
    # ground truth exactly.
    flat_imp_ref = full_imp_ref.reshape(-1)
    local_slice = flat_imp_ref[offsets[rank]: offsets[rank] + sizes[rank]].clone()
    param = FakeParam(local_slice.clone())

    class FakeFisher:
        saliency = 'fisher'
        def importance(self, name, p):
            return p.data.clone()

    named_params = {"model.layers.0.mlp.gate_proj.weight": param}
    named_shapes = {"model.layers.0.mlp.gate_proj.weight": (rows, cols)}

    mgr = GradualMaskManager(named_params, fsdp_model=None, prune_n=prune_n, prune_m=prune_m,
                              named_shapes=named_shapes)
    # Force the FSDP-aware branch even though fsdp_model=None here (test harness
    # doesn't build a real FSDP-wrapped model) -- monkeypatch the boolean the
    # same way `use_fsdp = _FSDP_AVAILABLE and fsdp_model is not None` would
    # evaluate to True in real training, by calling candidate_masks with a
    # truthy fsdp_model placeholder and patching _FSDP_AVAILABLE.
    import gmp_trainer
    gmp_trainer._FSDP_AVAILABLE = True

    fisher = FakeFisher()
    for sp in [0.1, 0.244, 0.392, 0.5]:
        new_masks = mgr.candidate_masks(fisher, sp, fsdp_model="not-none-sentinel")
        local_mask = new_masks["model.layers.0.mlp.gate_proj.weight"]
        local_dead = (~local_mask).sum().item()
        # ground truth: what SHOULD the dead count be for this rank's slice,
        # computed independently on the full un-sharded tensor
        gt_keep = mgr._nm_mask(full_imp_ref, torch.zeros(rows, cols, dtype=torch.bool), sp)
        gt_dead_local = (~gt_keep).reshape(-1)[offsets[rank]: offsets[rank] + sizes[rank]].sum().item()
        print(f"[rank {rank}] sparsity_req={sp:.3f} local_dead={local_dead} "
              f"expected_local_dead={gt_dead_local} local_shard_size={sizes[rank]} "
              f"{'OK' if local_dead == gt_dead_local else 'MISMATCH'}")
        # candidate_masks doesn't mutate self.masks -- feed forward for the next
        # sparsity level the same way update() would (current_mask accumulates).
        mgr.masks["model.layers.0.mlp.gate_proj.weight"] = local_mask

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
