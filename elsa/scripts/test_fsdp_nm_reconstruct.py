import os
import sys
import torch
import torch.distributed as dist

sys.path.insert(0, "/home1/doyoonkim/projects/elsa/lib")
from gmp_trainer import (
    _fsdp_gather_flat, _fsdp_scatter_flat,
    _fsdp_nm_reconstruct, _fsdp_nm_scatter_back,
    _pgd_nm_pre_target_2d, _pgd_nm_post_target_2d,
)


class FakeParam:
    def __init__(self, shape):
        self.shape = shape


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(0)

    rows, cols, prune_n, prune_m = 17, 20, 2, 4  # deliberately not evenly divisible by world_size
    full_imp = torch.rand(rows, cols)
    full_mask = torch.rand(rows, cols) > 0.5

    # simulate FSDP's arbitrary flat chunking: uneven split sizes, not row/col aligned
    total = rows * cols
    base = total // world_size
    sizes = [base] * world_size
    sizes[-1] += total - base * world_size  # remainder on last rank
    offsets = [sum(sizes[:r]) for r in range(world_size)]

    flat_imp = full_imp.reshape(-1)
    flat_mask = full_mask.reshape(-1)
    local_imp = flat_imp[offsets[rank]: offsets[rank] + sizes[rank]].clone()
    local_mask = flat_mask[offsets[rank]: offsets[rank] + sizes[rank]].clone()

    param_shape = (rows, cols)

    # ---- ground truth: run the SAME core function on the full tensor directly ----
    gt_pre = _pgd_nm_pre_target_2d(full_imp, full_mask, prune_n, prune_m, prune_m - prune_n)
    gt_post = _pgd_nm_post_target_2d(full_imp, prune_n, prune_m)
    gt_pre_local = gt_pre.reshape(-1)[offsets[rank]: offsets[rank] + sizes[rank]]
    gt_post_local = gt_post.reshape(-1)[offsets[rank]: offsets[rank] + sizes[rank]]

    # ---- reconstruct via gather, run same function, scatter back ----
    recon_imp, recon_mask = _fsdp_nm_reconstruct(local_imp, local_mask, param_shape)
    assert recon_imp.shape == (rows, cols), f"reconstruct shape wrong: {recon_imp.shape}"
    assert torch.allclose(recon_imp, full_imp), "reconstructed imp doesn't match original!"
    assert torch.equal(recon_mask, full_mask), "reconstructed mask doesn't match original!"

    pre_full = _pgd_nm_pre_target_2d(recon_imp, recon_mask, prune_n, prune_m, prune_m - prune_n)
    pre_local = _fsdp_nm_scatter_back(pre_full, local_imp.numel(), rank, local_imp.shape)
    assert torch.equal(pre_local, gt_pre_local), f"[rank {rank}] pre_target mismatch!"

    post_full = _pgd_nm_post_target_2d(recon_imp, prune_n, prune_m)
    post_local = _fsdp_nm_scatter_back(post_full, local_imp.numel(), rank, local_imp.shape)
    assert torch.equal(post_local, gt_post_local), f"[rank {rank}] post_target mismatch!"

    print(f"[rank {rank}] ALL CHECKS PASSED (local_size={sizes[rank]})")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
