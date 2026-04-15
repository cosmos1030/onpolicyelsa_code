# RAC SparseGPT Pruning — CUDA OOM Debug Log

Goal: run `do_train=False --prune --pruning_method SparseGPT` on a single GPU
and then evaluate the pruned model with lighteval+vLLM.

---

## Cluster / hardware

| Node type | VRAM |
|-----------|------|
| A100-80GB (partition=A100-80GB, qos=hpgpu) | ~80 GiB |
| A100-40GB-PCIe (partition=A100-40GB-PCIe, qos=normal) | ~39.49 GiB |

---

## Attempt log

### Job 334120 — A100-80GB, expandable_segments
- Config: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, A100-80GB
- **Result: FAIL** — `torch.OutOfMemoryError: Tried to allocate 20 MiB, 78 GiB free`
  during `super().__init__()` NCCL/accelerate init corrupts allocator.
- Root cause: NCCL initialised via SLURM env vars → corrupts CUDA allocator state.
- Fix attempted: unset SLURM_LOCALID etc. + move model after super().__init__()

### Jobs 334499, 334503, 334542 — A100-80GB, expandable_segments
- Config: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, A100-80GB
- **Result: FAIL** — `RuntimeError: CUDA driver error: out of memory` at `model.to()`
- Root cause: `expandable_segments` requires CUDA VMM API (`cuMemCreate`) which
  this cluster's CUDA driver does not support.
- **DO NOT USE `expandable_segments:True` on this cluster — ever.**

### Job 335067 — A100-80GB, no expandable_segments
- Config: default allocator, A100-80GB
- **Result: FAIL** — same CUDA driver OOM at model.to()
- Root cause: A100-80GB nodes fail at model.to() regardless of allocator setting.
  (Possibly 80GB nodes have a different driver/firmware that corrupts the allocator
   after accelerate/NCCL distributed init.)
- **DO NOT USE A100-80GB nodes for prune-only runs.**

### Job 335087 — A100-40GB-PCIe, device_map={"": 0}
- Config: `model_utils.py` with `device_map={"": 0}` in from_pretrained
- **Result: FAIL** — `unable to mmap 1503300328 bytes: Cannot allocate memory`
  (CPU-side safetensors mmap failure, not GPU OOM)
- Root cause: safetensors mmap for ~1.4 GB file fails, likely due to ulimit/
  huge-page limitations on the node.
- **DO NOT USE `device_map={"": 0}` in from_pretrained.**

### Jobs 335139, 335142 — A100-40GB-PCIe, expandable_segments
- Config: `expandable_segments:True`, A100-40GB-PCIe
- **Result: FAIL** — `CUDA driver error: out of memory` at model.to()
- Same driver VMM limitation as on 80GB nodes.
- **DO NOT USE `expandable_segments:True` on any node in this cluster.**

### Job 335145 — A100-40GB-PCIe, default allocator ← BEST SO FAR
- Config: no PYTORCH_CUDA_ALLOC_CONF set, A100-40GB-PCIe
- **Result: PARTIAL** — model.to() succeeds (1.22 GB used, 40.7 GB free)
  but SparseGPT Hessian allocation fails.
- Error: `Tried to allocate 36.00 MiB, 36.72 GiB free` at first `down_proj` H
- Root cause: model.to() allocates hundreds of separate small CUDA tensors,
  fragmenting GPU VA space. All small H matrices (q/k/v/gate/up 4 MB each)
  allocate fine. First `down_proj` H = 36 MB (cols=3072, float32 3072×3072×4)
  fails because no contiguous 36 MB VA region exists.
  ~28 blocks × 6 small layers = 168 H matrices (~1 GB) allocated before failure.
- Fix: allocate H on CPU, move to GPU only in fasterprune. → next attempt.

### Job 335204 — A100-40GB-PCIe, H on CPU (cancelled but ran anyway)
- Config: H matrices allocated on `device='cpu'` in sparsegpt.py `__init__`
- **Result: FAIL** — `DefaultCPUAllocator: can't allocate memory: 37748736 bytes, Error 12 (ENOMEM)`
- Root cause: SLURM default CPU memory limit per job is too low. The process
  uses almost all of it loading dataset/tokenizer/etc.; 36 MB CPU alloc fails.
- Fix: add `--mem=32G` to SLURM script AND use GPU-only approach.
- **DO NOT allocate H on CPU** — CPU RAM is also limited.

### Job 335146 — A100-40GB-PCIe, backend:cudaMallocAsync
- Config: `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`, A100-40GB-PCIe
- **Result: FAIL** — `RuntimeError: CUDA error: out of memory` at model.to()
- Root cause: cudaMallocAsync stream-ordered allocator also fails at model.to()
  on this cluster (same driver incompatibility).
- **DO NOT USE `backend:cudaMallocAsync` on this cluster.**

---

## Current fix (in code as of 2026-04-14)

### sparsegpt.py
- `__init__`: H allocated on `device='cpu'` instead of GPU
- `add_batch`: compute `inp @ inp.t()` on GPU (fast), move result to CPU via
  `.to(self.H.device)` (one PCIe transfer per batch, max 36 MB for down_proj)
- `fasterprune`: move H from CPU→GPU at start (`self.H = self.H.to(self.dev)`)

### grpo_trainer.py
- `_move_model_to_device` patched to no-op for prune-only runs
- Explicit `self.model.to("cuda:0")` + `torch.cuda.empty_cache()` before pruning
- `infer_auto_device_map` still called (needed for model.to() to work on this cluster)

### slurm script
- Partition: A100-40GB-PCIe, qos=normal
- No PYTORCH_CUDA_ALLOC_CONF set (default caching allocator)
- SLURM distributed env vars unset

---

## What to try if current fix also fails

1. Reduce `prune_calib_tokens` to 5000 to speed up debugging
2. Check if the H CPU→GPU move in fasterprune itself triggers a new OOM
   (it moves 36 MB one layer at a time, so peak GPU H = 36 MB — should be fine)
3. If calibration is too slow due to PCIe transfers: buffer H updates on GPU in
   a small temporary tensor, flush to CPU every N batches
