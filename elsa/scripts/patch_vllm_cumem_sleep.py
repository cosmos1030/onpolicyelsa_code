#!/usr/bin/env python3
"""Patch vllm==0.10.0's device_allocator/cumem.py with two fixes for GPU
memory corruption observed when vLLM's sleep-mode CuMemAllocator shares one
GPU/CUDA context with another live PyTorch allocator (e.g. a training model
co-located via a small gpu_memory_utilization fraction, as in ELSA's OPKD
setup). Idempotent -- safe to run multiple times or after a fresh vllm
reinstall on any server; it edits whatever `vllm` package `python -m
scripts.patch_vllm_cumem_sleep` (or plain `python patch_vllm_cumem_sleep.py`)
resolves via `import vllm`, not a hardcoded path.

Usage: /path/to/env/bin/python patch_vllm_cumem_sleep.py

Fix 1 (upstream, vllm-project/vllm#23477, merged 2025-08-24, in vllm>=0.10.2):
CuMemAllocator.__init__ passed bound methods directly as the pluggable
allocator's malloc/free callbacks without keeping a strong reference to them.
CPython can garbage-collect those ephemeral bound-method objects; when the
C-extension later calls back into a freed one, symptoms are a sporadic "CUDA
error: an illegal memory access" inside vLLM's own code, or an
interpreter-level "Fatal Python error: none_dealloc: deallocating None" in
an unrelated thread, both at random points (seen at steps ranging ~200 to
~1650) after many sleep()/wake_up() cycles. Backported here because
upgrading to vllm>=0.10.2 breaks lighteval compatibility (no released
lighteval version supports vllm>=0.10.2 as of 2026-08).

Fix 2 (not upstream -- ELSA-local): CuMemAllocator.sleep() calls
gc.collect() + torch.cuda.empty_cache() unconditionally at the end, which
operates on the process-wide default PyTorch caching allocator, not a pool
scoped to this allocator's own tensors. When this engine shares a GPU/CUDA
context with another live PyTorch allocator, unmap_and_release() (releasing
vLLM's own pool) and empty_cache() (releasing the OTHER allocator's cache)
both hand virtual memory back to the CUDA driver independently, with no
coordination between the two allocators sharing that address space --
observed as a rare (~1/14 runs), random-onset segfault or illegal-memory-
access shortly after a wake()/sleep() cycle, surfacing later (e.g. inside an
unrelated torch.kl_div call) because CUDA errors report asynchronously.
Adding torch.cuda.synchronize() right before gc.collect()/empty_cache() is a
pure wait -- it cannot change results, only close the race window by
guaranteeing the other allocator's pending CUDA work is done before we free.

Fix 3 (not upstream -- ELSA-local): symmetric counterpart to Fix 2, on the
wake_up() side. wake_up() does a raw driver-level cuMemMap/cuMemSetAccess
(create_and_map()) and a raw libcudart.cudaMemcpy() on bare pointers -- like
sleep()'s unmap_and_release()/empty_cache(), neither goes through PyTorch's
caching allocator or its stream bookkeeping, so neither is ordered against
the co-located training model's in-flight kernels on the same GPU/CUDA
context. Bracketing the wake_up() loop with torch.cuda.synchronize() closes
the race window on both sides: the pre-sync ensures no training kernel is
still touching memory while the driver remaps address space back in, and the
post-sync ensures the raw memcpy has actually finished before the caller
resumes issuing training-side kernels that may depend on it. Found
2026-08-24 after Fix 2 alone did not fully eliminate a residual, sporadic
(~1/9 runs) segfault -- e.g. surfacing later inside an unrelated
torch.nn.functional.kl_div call, consistent with async-reported GPU memory
corruption rather than a Python-level bug.

All three patches are no-ops for a plain single-allocator vLLM deployment (a
normal vLLM server owning its whole GPU) -- they only matter for this
co-located, memory-shared setup.
"""
import re
import sys


def main() -> int:
    import vllm.device_allocator.cumem as cumem_mod
    path = cumem_mod.__file__
    src = open(path).read()
    changed = False

    # --- Fix 1: strong references to the malloc/free callbacks ---
    if "self.python_malloc_callback = self._python_malloc_callback" in src:
        print(f"[fix 1] already applied in {path}")
    else:
        old_init_tail = (
            "        self.pointer_to_data: dict[int, AllocationData] = {}\n"
            "        self.current_tag: str = CuMemAllocator.default_tag\n"
            "        self.allocator_and_pools: dict[str, Any] = {}\n"
            "\n"
            "    def python_malloc_callback(self, allocation_handle: HandleType) -> None:\n"
        )
        new_init_tail = (
            "        self.pointer_to_data: dict[int, AllocationData] = {}\n"
            "        self.current_tag: str = CuMemAllocator.default_tag\n"
            "        self.allocator_and_pools: dict[str, Any] = {}\n"
            "        # Creating strong references to the two callbacks here to prevent\n"
            "        # these ephemeral bound-method objects being garbage collected.\n"
            "        # See discussions in https://github.com/vllm-project/vllm/pull/22724\n"
            "        # (backported via scripts/patch_vllm_cumem_sleep.py -- upstream fix\n"
            "        # is vllm-project/vllm#23477, merged 2025-08-24, first released in\n"
            "        # 0.10.2; not usable here due to lighteval incompatibility.)\n"
            "        self.python_malloc_callback = self._python_malloc_callback\n"
            "        self.python_free_callback = self._python_free_callback\n"
            "\n"
            "    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:\n"
        )
        if old_init_tail not in src:
            print(f"[fix 1] FAILED: expected __init__ text not found in {path} "
                  "-- vllm version/file layout may differ from what this patch targets "
                  "(written against vllm==0.10.0). Aborting without changes.",
                  file=sys.stderr)
            return 1
        src = src.replace(old_init_tail, new_init_tail, 1)

        old_free_def = "    def python_free_callback(self, ptr: int) -> HandleType:\n"
        if src.count(old_free_def) != 1:
            print(f"[fix 1] FAILED: expected python_free_callback def not found "
                  f"(or not unique) in {path}. Aborting without changes.", file=sys.stderr)
            return 1
        src = src.replace(old_free_def, "    def _python_free_callback(self, ptr: int) -> HandleType:\n", 1)
        changed = True
        print(f"[fix 1] applied to {path}")

    # --- Fix 2: synchronize() before sleep()'s gc.collect()/empty_cache() ---
    if re.search(r"torch\.cuda\.synchronize\(\)\s*\n\s*gc\.collect\(\)\s*\n\s*torch\.cuda\.empty_cache\(\)", src):
        print(f"[fix 2] already applied in {path}")
    else:
        old_tail = (
            "            unmap_and_release(handle)\n"
            "\n"
            "        gc.collect()\n"
            "        torch.cuda.empty_cache()\n"
        )
        new_tail = (
            "            unmap_and_release(handle)\n"
            "\n"
            "        # Manually patched (not upstream, see scripts/patch_vllm_cumem_sleep.py):\n"
            "        # empty_cache() below operates on the process-wide default PyTorch\n"
            "        # caching allocator, not a pool scoped to this allocator's own tensors --\n"
            "        # racy when this engine shares a GPU/CUDA context with another live\n"
            "        # PyTorch allocator. synchronize() is a pure wait, cannot change results.\n"
            "        torch.cuda.synchronize()\n"
            "        gc.collect()\n"
            "        torch.cuda.empty_cache()\n"
        )
        if old_tail not in src:
            print(f"[fix 2] FAILED: expected sleep() tail text not found in {path} "
                  "-- vllm version/file layout may differ from what this patch targets. "
                  "Aborting without changes.", file=sys.stderr)
            return 1
        src = src.replace(old_tail, new_tail, 1)
        changed = True
        print(f"[fix 2] applied to {path}")

    # --- Fix 3: synchronize() bracketing wake_up()'s create_and_map/cudaMemcpy loop ---
    if re.search(r"def wake_up.*?\n(?:.*\n){0,6}?\s*torch\.cuda\.synchronize\(\)\s*\n\s*for ptr, data in self\.pointer_to_data\.items\(\):", src, re.DOTALL):
        print(f"[fix 3] already applied in {path}")
    else:
        old_wake_head = (
            "        for ptr, data in self.pointer_to_data.items():\n"
            "            if tags is None or data.tag in tags:\n"
            "                handle = data.handle\n"
            "                create_and_map(handle)\n"
            "                if data.cpu_backup_tensor is not None:\n"
            "                    cpu_backup_tensor = data.cpu_backup_tensor\n"
            "                    if cpu_backup_tensor is not None:\n"
            "                        size_in_bytes = cpu_backup_tensor.numel(\n"
            "                        ) * cpu_backup_tensor.element_size()\n"
            "                        cpu_ptr = cpu_backup_tensor.data_ptr()\n"
            "                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)\n"
            "                        data.cpu_backup_tensor = None\n"
        )
        if src.count(old_wake_head) != 1:
            print(f"[fix 3] FAILED: expected wake_up() body text not found (or not unique) "
                  f"in {path}. Aborting without changes.", file=sys.stderr)
            return 1
        new_wake_head = (
            "        # Manually patched (not upstream, see scripts/patch_vllm_cumem_sleep.py,\n"
            "        # Fix 3): symmetric counterpart to sleep()'s synchronize(). Neither\n"
            "        # create_and_map() (raw cuMemMap) nor the raw cudaMemcpy below go\n"
            "        # through PyTorch's caching allocator or stream bookkeeping, so\n"
            "        # neither is ordered against the co-located training model's\n"
            "        # in-flight kernels on the same GPU/CUDA context.\n"
            "        torch.cuda.synchronize()\n"
            "        for ptr, data in self.pointer_to_data.items():\n"
            "            if tags is None or data.tag in tags:\n"
            "                handle = data.handle\n"
            "                create_and_map(handle)\n"
            "                if data.cpu_backup_tensor is not None:\n"
            "                    cpu_backup_tensor = data.cpu_backup_tensor\n"
            "                    if cpu_backup_tensor is not None:\n"
            "                        size_in_bytes = cpu_backup_tensor.numel(\n"
            "                        ) * cpu_backup_tensor.element_size()\n"
            "                        cpu_ptr = cpu_backup_tensor.data_ptr()\n"
            "                        libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)\n"
            "                        data.cpu_backup_tensor = None\n"
            "        torch.cuda.synchronize()\n"
        )
        src = src.replace(old_wake_head, new_wake_head, 1)
        changed = True
        print(f"[fix 3] applied to {path}")

    if changed:
        open(path, "w").write(src)

    # Sanity check: make sure the patched file still imports cleanly.
    import importlib
    importlib.reload(cumem_mod)
    print("import OK after patch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
