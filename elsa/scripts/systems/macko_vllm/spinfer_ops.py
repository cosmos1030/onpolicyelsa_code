"""SpInfer (EuroSys'25, TCA-BME format) for batched sparse decode inside vLLM.

compress(w)  -- GPU, pure torch: builds exactly the arrays InitSparseMatrixA_bitmap_v6 builds
                (64x64 global tiles -> 16x64 median tiles -> 2x2 groups of 8x8 tiles, column-major
                within a group; 64-bit occupancy bitmap per 8x8 tile; values in traversal order,
                each global tile zero-padded to a multiple of 8).
spmm(x, c)   -- y = x @ W^T for x [n,K] fp16, any n: padded to 8/16/32/64/128 or a multiple of 128.
"""
import math
import os

import torch

SPINFER_DIR = os.environ.get("SPINFER_DIR", "/home1/doyoonkim/projects/third_party_kernels/spinfer")
_loaded = False


def load():
    global _loaded
    if _loaded:
        return
    from torch.utils.cpp_extension import load as _load
    here = os.path.dirname(os.path.abspath(__file__))
    cap = torch.cuda.get_device_capability()
    bdir = os.environ.get("SPINFER_BUILD_DIRECTORY",
                          os.path.expanduser(f"~/.cache/spinfer_build_sm{cap[0]}{cap[1]}"))
    os.makedirs(bdir, exist_ok=True)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{cap[0]}.{cap[1]}")
    _load(name="spinfer_ext", sources=[os.path.join(here, "spinfer_ext.cu")],
          extra_include_paths=[os.path.join(SPINFER_DIR, "csrc")],
          extra_cuda_cflags=["-O3", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                             "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
          build_directory=bdir, is_python_module=False, verbose=False)
    _loaded = True


def _perm(device):
    # position (row*64+col) inside a 64x64 global tile, in SpInfer's traversal order
    idx = []
    for mm in range(4):                    # median tiles (16 rows each; one 64-wide median tile per row band)
        for gk in (0, 2, 4, 6):            # 2x2 groups of 8x8 tiles (a 16-row band holds one group row)
            for j in range(2):             # column-major inside the group
                for i in range(2):
                        r0 = mm * 16 + i * 8
                        c0 = (gk + j) * 8
                        for ro in range(8):
                            for co in range(8):
                                idx.append((r0 + ro) * 64 + c0 + co)
    return torch.tensor(idx, dtype=torch.long, device=device)


_PERM = {}


@torch.no_grad()
def compress(w):
    """w: [M,K] fp16 cuda -> (vals, tog, tom, bitmap, maxnnz) cuda tensors."""
    M, K = w.shape
    assert M % 64 == 0 and K % 64 == 0, (M, K)
    dev = w.device
    if dev not in _PERM:
        _PERM[dev] = _perm(dev)
    gM, gK = M // 64, K // 64
    G = gM * gK
    t = w.to(torch.float16).reshape(gM, 64, gK, 64).permute(0, 2, 1, 3).reshape(G, 4096)
    t = t[:, _PERM[dev]]                                     # [G,4096] traversal order
    nz = t != 0
    cnt = nz.sum(1, dtype=torch.int32)                      # per global tile
    pad = (cnt + 7) // 8 * 8
    tog = torch.zeros(G + 1, dtype=torch.int32, device=dev)
    tog[1:] = torch.cumsum(pad, 0, dtype=torch.int32)
    med = nz.view(G, 4, 1024).sum(2, dtype=torch.int32)
    tom = torch.zeros(G, 4, dtype=torch.int32, device=dev)
    tom[:, 1:] = torch.cumsum(med[:, :3], 1, dtype=torch.int32)
    shifts = torch.arange(64, device=dev, dtype=torch.int64)
    bitmap = (nz.view(G * 64, 64).to(torch.int64) << shifts).sum(1)   # distinct bits: sum == OR
    total = int(tog[-1].item())
    vals = torch.zeros(max(total, 8), dtype=torch.float16, device=dev)
    rank = torch.cumsum(nz, 1, dtype=torch.int32) - 1                 # rank of each nnz in its tile
    pos = (tog[:-1].unsqueeze(1) + rank)[nz].long()
    vals[pos] = t[nz]
    mx = int(pad.max().item()) if G else 0
    mx = (mx + 63) // 64 * 64
    maxnnz = torch.tensor([mx], dtype=torch.int32, device=dev)
    del t, nz, rank, pos
    return vals, tog, tom.reshape(-1).contiguous(), bitmap, maxnnz


def pad_n(n):
    for b in (8, 16, 32, 64, 128):
        if n <= b:
            return b
    return (n + 127) // 128 * 128


def spmm(x, c, M, split_k=1):
    load()
    n = x.shape[0]
    np_ = pad_n(n)
    if np_ != n:
        x = torch.nn.functional.pad(x, (0, 0, 0, np_ - n))
    y = torch.ops.spinfer.spmm(x.contiguous(), c[0], c[1], c[2], c[3], c[4], M, split_k)
    return y[:n] if np_ != n else y


_SK_CACHE = {}


@torch.no_grad()
def tune_split_k(c, M, K, ns=(8, 16, 32, 64, 128), iters=20):
    """Best Split_K per padded batch for this (M,K) shape; cached per shape (all layers of a
    shape share it). Only values that divide the K-block count are tried."""
    load()
    key = (M, K)
    if key in _SK_CACHE:
        return _SK_CACHE[key]
    nkb = K // 64
    cands = [s for s in range(1, 9) if nkb % s == 0]
    best = {}
    for n in ns:
        x = torch.randn(n, K, device="cuda", dtype=torch.float16)
        tbest = None
        for s in cands:
            for _ in range(3):
                torch.ops.spinfer.spmm(x, c[0], c[1], c[2], c[3], c[4], M, s)
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            for _ in range(iters):
                torch.ops.spinfer.spmm(x, c[0], c[1], c[2], c[3], c[4], M, s)
            e1.record()
            torch.cuda.synchronize()
            t = e0.elapsed_time(e1)
            if tbest is None or t < tbest:
                tbest, best[n] = t, s
    _SK_CACHE[key] = best
    return best
