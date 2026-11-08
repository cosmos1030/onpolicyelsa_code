"""_pgd_kth_smallest (FSDP, GMP_PGD_BRACKET_FSDP=1 bracket path and radix fallback) must return
exactly the global r-th smallest value over all ranks' shards. torchrun --nproc_per_node=2."""
import os, sys, torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from absl import flags
flags.FLAGS(['x'])
os.environ['GMP_PGD_BRACKET_FSDP'] = '1'
from lib.gmp_trainer import _pgd_kth_smallest, _PGD_KTH_STATS

dist.init_process_group('nccl')
rank = dist.get_rank(); torch.cuda.set_device(rank); dev = torch.device('cuda', rank)
g = torch.Generator(device='cpu').manual_seed(0)
cases = {
    'lognormal': lambda n: torch.randn(n, generator=g).exp(),
    'heavy ties': lambda n: torch.randint(0, 50, (n,), generator=g).float(),
    'log importance w/ sentinel': lambda n: torch.where(torch.rand(n, generator=g) < 0.3, torch.full((n,), 1e30),
                                                         torch.randn(n, generator=g)),
}
bad = 0
for name, mk in cases.items():
    # uneven shards per rank, several tensors each (like FSDP shards of many weights)
    shards = [mk(int(n)).to(dev) for n in ([300_000, 7, 1_200_000, 0, 55_555] if rank == 0 else [10, 900_001, 333_333])]
    allv = [torch.empty(0)] * dist.get_world_size()
    flat = torch.cat(shards).cpu()
    dist.all_gather_object(allv, flat)
    ref_all = torch.cat(allv).sort().values
    n = ref_all.numel()
    for r in [1, 2, n // 3, n // 2, n - 1, n, 12345]:
        got = _pgd_kth_smallest(lambda: iter(shards), r, dev, True)
        ref = ref_all[r - 1].item()
        if got != ref:
            bad += 1
            print(f'[rank{rank}] MISMATCH {name} r={r}: got {got} ref {ref}', flush=True)
if rank == 0:
    print(f'bracket hits/misses {_PGD_KTH_STATS[0]}/{_PGD_KTH_STATS[1]}  mismatches={bad}', flush=True)
    print('BRACKET_OK' if bad == 0 else 'BRACKET_FAIL', flush=True)
dist.destroy_process_group()
