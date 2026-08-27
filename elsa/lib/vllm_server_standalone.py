"""Standalone vLLM server entry-point for FSDP+OPKD.

Launched via subprocess.Popen (a fresh `python` exec, NOT multiprocessing.Process)
so it is a genuinely independent OS process: no inherited CUDA context, no
inherited torch.distributed/NCCL state, no torchrun env vars (RANK/WORLD_SIZE/
MASTER_ADDR/...) unless explicitly passed. vLLM's own distributed runtime (its
TP worker processes + their NCCL/gloo group) then bootstraps from a completely
clean process tree instead of nesting inside torchrun's -- see
lib/vllm_proc.py's launch_vllm_server() docstring for why the previous
multiprocessing.Process-based approach (daemon/fork/rendezvous bugs) is gone.

Talks to its one client (the training process) over a Unix-domain-socket
multiprocessing.connection -- same wire protocol (plain send/recv of pickled
Python objects) the old Queue-based version used, just not tied to a
parent/child relationship.
"""
import argparse
import os
import pathlib
import sys

# Must match lib/vllm_proc.py's setting -- see the comment there. Both sides
# of the connection need 'file_system' or the receiver's first tensor-bearing
# recv() raises AuthenticationError trying to fetch a shared-memory fd back
# from a sender it has no multiprocessing parent/child relationship with.
import torch.multiprocessing as _torch_mp
_torch_mp.set_sharing_strategy('file_system')


def _load_weights_on_worker(worker_self, items):
    """Runs inside each vLLM TP worker via collective_rpc -- every worker gets
    the same full (unsharded) items list; each parameter's own weight_loader
    slices its shard by tp_rank, exactly like the normal from-HF-checkpoint
    load path."""
    worker_self.model_runner.model.load_weights(items)


def _sync_weights_to_engine(engine, state_dict, tensor_parallel_size):
    """TP=1: reaches the single driver_worker directly. TP>1: broadcasts via
    collective_rpc so every TP worker re-shards the same full state_dict
    through its own weight_loader -- a plain driver_worker update would only
    patch rank 0's shard and leave the rest of the TP group stale."""
    items = list(state_dict.items())
    if tensor_parallel_size > 1:
        engine.collective_rpc(_load_weights_on_worker, args=(items,))
    else:
        # vLLM 0.10+: model_executor lives under engine_core
        executor = (engine.llm_engine.engine_core.model_executor
                    if hasattr(engine.llm_engine, 'engine_core')
                    else engine.llm_engine.model_executor)
        executor.driver_worker.model_runner.model.load_weights(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--cuda-devices', required=True)
    ap.add_argument('--tp-size', type=int, default=1)
    ap.add_argument('--gpu-mem', type=float, default=0.15)
    ap.add_argument('--max-len', type=int, default=768)
    ap.add_argument('--enforce-eager', action='store_true')
    ap.add_argument('--address', required=True, help='Unix socket path')
    ap.add_argument('--authkey', required=True)
    args = ap.parse_args()

    # args.cuda_devices is a RELATIVE index (or comma-list) computed by
    # main.py as if CUDA_VISIBLE_DEVICES were unset (e.g. "0" for
    # gmp_opkd_vllm_gpu_index=0). Map it through whatever CUDA_VISIBLE_DEVICES
    # this subprocess already inherited from its parent -- blindly overwriting
    # with the raw relative index (the old behavior) silently re-expands
    # visibility to ALL physical GPUs and always lands on physical GPU0,
    # which is invisible/harmless for a single job with no outer restriction
    # (SLURM jobs, or this container running one job at a time) but collides
    # multiple concurrent jobs' vLLM sidecars onto the same physical GPU0
    # when the outer script partitions this container's 4 GPUs across jobs
    # via CUDA_VISIBLE_DEVICES (e.g. b200_scripts/run_gmp_pgd_klgate_8b_fsdp2gpu_parallel.sh)
    # -- reproduced live 2026-08-27: two concurrent 8B FSDP jobs' vLLM
    # sidecars both landed on physical GPU0, OOMing a third process there.
    _parent_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if _parent_cvd:
        _physical = [d.strip() for d in _parent_cvd.split(',') if d.strip()]
        _selected = [_physical[int(i)] for i in args.cuda_devices.split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(_selected)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    os.environ.setdefault('VLLM_USE_V1', '0')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('VLLM_HOST_IP', '127.0.0.1')
    if args.tp_size > 1:
        # vLLM's own internal TP-worker rendezvous (torch.distributed.
        # init_process_group, separate from ANY outer training pg) resolves
        # its rendezvous hostname via getaddrinfo() when MASTER_ADDR/PORT
        # aren't set. On this cluster that resolves to "localhost.localdomain"
        # and fails (errno 97, [c10d] socket.cpp warning) -- harmless when it's
        # the only c10d rendezvous on the box (falls back fine), but hangs
        # forever if a second, unrelated c10d group (e.g. the FSDP trainer's)
        # is concurrently alive on the same physical GPUs, most likely due to
        # both falling back to the same abstract-namespace socket/store name.
        # Force explicit loopback + a private port so vLLM's own rendezvous
        # never needs that fallback path at all.
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ.setdefault('MASTER_PORT', str(20000 + (os.getpid() % 10000)))
        # vLLM's mp executor defaults VLLM_WORKER_MULTIPROC_METHOD to 'fork'.
        # This process's own driver worker touches CUDA before vLLM forks the
        # extra TP-rank worker(s), and a fork of a CUDA-initialized process
        # crashes ("Cannot re-initialize CUDA in forked subprocess") -- force
        # spawn so those workers get a clean CUDA context of their own.
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    else:
        # TP=1: vLLM still calls torch.distributed.init_process_group()
        # internally for its single self-contained worker. Same hostname-
        # resolution rendezvous problem as the TP>1 case above -- it hangs
        # if another live c10d group (e.g. the FSDP trainer's) exists on the
        # same physical GPU(s), which explicit MASTER_ADDR/PORT alone did
        # NOT fix for TP>1 (still hung -- see repro_nested_nccl_test2.py).
        # What DOES work: pre-initialize a trivial world_size=1 gloo pg of
        # our own first -- vLLM checks torch.distributed.is_initialized()
        # and skips its own rendezvous entirely when it's already true, so
        # its real init never happens and there's nothing to hang on.
        import uuid
        import torch.distributed as _td
        _pg_file = f'/tmp/vllm_pg_{uuid.uuid4().hex}'
        if not _td.is_initialized():
            _td.init_process_group(
                backend='gloo',
                init_method=f'file://{_pg_file}',
                world_size=1,
                rank=0,
            )
    tc = pathlib.Path.home() / '.triton_cache_vllm_standalone'
    tc.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(tc)

    from vllm import LLM, SamplingParams

    engine = LLM(
        args.model,
        dtype='bfloat16',
        gpu_memory_utilization=args.gpu_mem,
        trust_remote_code=True,
        max_model_len=args.max_len,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=args.tp_size,
    )

    from multiprocessing.connection import Listener
    listener = Listener(args.address, family='AF_UNIX', authkey=args.authkey.encode())
    # Printed to stdout, which the launcher redirects to a log file it polls
    # for this exact marker -- see launch_vllm_server()'s readiness wait.
    print('VLLM_SERVER_READY', flush=True)
    conn = listener.accept()
    try:
        while True:
            req = conn.recv()
            if req is None:
                conn.send('done')
                break
            if isinstance(req, tuple) and req[0] == 'sync_weights':
                _, state_dict = req
                try:
                    _sync_weights_to_engine(engine, state_dict, args.tp_size)
                    conn.send('synced')
                except Exception as e:
                    conn.send(f'sync_error: {e}')
                continue
            prompt_ids_list, max_new, temp = req
            params = SamplingParams(max_tokens=max_new, temperature=temp, top_p=0.95)
            inputs = [{'prompt_token_ids': ids} for ids in prompt_ids_list]
            outputs = engine.generate(inputs, params)
            conn.send([out.outputs[0].token_ids for out in outputs])
    finally:
        conn.close()
        listener.close()


if __name__ == '__main__':
    main()
