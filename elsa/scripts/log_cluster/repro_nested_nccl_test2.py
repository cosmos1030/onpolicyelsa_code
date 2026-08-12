"""Second isolation variable: does ORDER matter -- vLLM TP=2's own pg created
BEFORE the training pg (instead of after, which we already reproduced hanging)?
No dist.init_process_group() call exists yet when rank 0 launches vLLM here;
rank 1 just sleeps long enough for that launch to finish, then both ranks call
dist.init_process_group() together for the "training" pg on the SAME GPUs.

Usage: CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
    scripts/log_cluster/repro_nested_nccl_test2.py
"""
import datetime
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    if local_rank == 0:
        from lib.vllm_proc import launch_vllm_server
        print("[rank 0] launching standalone vLLM TP=2 server BEFORE any "
              "training dist.init_process_group() call...", flush=True)
        adapter = launch_vllm_server(
            'Qwen/Qwen3-8B',
            cuda_device_str='0,1',
            gpu_mem=0.15,
            max_len=768,
            enforce_eager=True,
            default_max_new=8,
            default_temp=0.7,
            startup_timeout=180,
            tensor_parallel_size=2,
        )
        print("[rank 0] vLLM server READY (pre-training-pg)", flush=True)
        out = adapter.generate([{'prompt_token_ids': [9707, 11, 1246]}])
        print("[rank 0] generate result:", out, flush=True)
        # signal rank 1 that vLLM launch is done
        with open('/tmp/repro2_vllm_ready.flag', 'w') as f:
            f.write('ready')
    else:
        print(f"[rank {local_rank}] waiting for rank 0's vLLM launch (polling flag file)...", flush=True)
        while not os.path.exists('/tmp/repro2_vllm_ready.flag'):
            time.sleep(2)
        print(f"[rank {local_rank}] flag seen, proceeding to init_process_group", flush=True)

    print(f"[rank {local_rank}] now calling dist.init_process_group() for the "
          f"training pg (same physical GPUs vLLM just used)...", flush=True)
    dist.init_process_group(
        backend='nccl',
        device_id=torch.device(f'cuda:{local_rank}'),
        timeout=datetime.timedelta(minutes=2),
    )
    print(f"[rank {local_rank}] training NCCL pg initialized (world_size={world_size})", flush=True)
    dist.barrier()
    print(f"[rank {local_rank}] barrier passed -- test PASSED", flush=True)

    if local_rank == 0:
        adapter.shutdown()
        print("[rank 0] vLLM server shut down cleanly", flush=True)
        os.remove('/tmp/repro2_vllm_ready.flag')


if __name__ == '__main__':
    main()
