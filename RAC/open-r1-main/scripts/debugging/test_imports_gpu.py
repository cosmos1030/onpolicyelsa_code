import os
import sys

print("PYTORCH_CUDA_ALLOC_CONF before imports:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "NOT SET"))

import torch
print("After torch:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "NOT SET"))

from transformers import AutoModelForCausalLM
print("After transformers:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "NOT SET"))

import accelerate
print("After accelerate:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "NOT SET"))

try:
    import vllm
    print("After vllm:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "NOT SET"))
except Exception as e:
    print(f"vllm import error: {e}")

try:
    from trl import GRPOConfig
    print("After trl:", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "NOT SET"))
except Exception as e:
    print(f"trl import error: {e}")

print("\n--- GPU test after all imports ---")
torch.cuda.init()
free, total = torch.cuda.mem_get_info(0)
print(f"GPU: {free/1e9:.2f} GB free / {total/1e9:.2f} GB total")
print(f"Non-PyTorch GPU memory already used: {(total-free)/1e9:.3f} GB")

# Check if expandable_segments is enabled
try:
    stats = torch.cuda.memory_stats(0)
    seg_type = "unknown"
    print(f"allocator settings check: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")
except Exception as e:
    print(f"stats error: {e}")

for size_mb in [20, 100, 374, 500, 1000, 2000]:
    try:
        t = torch.zeros(size_mb * 1024 * 256, dtype=torch.float32, device="cuda:0")
        used = torch.cuda.memory_allocated(0) / 1024**2
        print(f"  {size_mb} MiB: OK (total allocated={used:.0f} MiB)")
        del t
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  {size_mb} MiB: FAILED -> {e}")
        break

print("Done")
