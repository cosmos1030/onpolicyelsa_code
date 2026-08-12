"""Isolates one variable: does an already-initialized (but idle/barrier-blocked)
torchrun NCCL process group on GPUs X,Y cause the standalone vLLM TP=2 server
(subprocess.Popen, fully independent process) to hang when it ALSO targets
GPUs X,Y -- as opposed to the torchrun-nesting we already fixed.

Usage (from elsa/): CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
    scripts/log_cluster/repro_nested_nccl_test.py
"""
import datetime
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend='nccl',
        device_id=torch.device(f'cuda:{local_rank}'),
        timeout=datetime.timedelta(minutes=2),
    )
    print(f"[rank {local_rank}] NCCL pg initialized (world_size={world_size})", flush=True)

    if local_rank == 0:
        from lib.vllm_proc import launch_vllm_server
        print("[rank 0] launching standalone vLLM TP=2 server on the SAME "
              "physical GPUs this NCCL pg is using...", flush=True)
        adapter = launch_vllm_server(
            'Qwen/Qwen3-8B',
            cuda_device_str='0,1',  # local indices == the same physical GPUs as this pg
            gpu_mem=0.15,
            max_len=768,
            enforce_eager=True,
            default_max_new=8,
            default_temp=0.7,
            startup_timeout=180,
            tensor_parallel_size=2,
        )
        print("[rank 0] vLLM server READY", flush=True)
        out = adapter.generate([{'prompt_token_ids': [9707, 11, 1246]}])
        print("[rank 0] generate result:", out, flush=True)
        adapter.shutdown()
        print("[rank 0] vLLM server shut down cleanly", flush=True)

    dist.barrier()
    print(f"[rank {local_rank}] barrier passed -- test PASSED", flush=True)


if __name__ == '__main__':
    main()
