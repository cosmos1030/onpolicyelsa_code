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

Both patches are no-ops for a plain single-allocator vLLM deployment (a
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

    if changed:
        open(path, "w").write(src)

    # Sanity check: make sure the patched file still imports cleanly.
    import importlib
    importlib.reload(cumem_mod)
    print("import OK after patch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
