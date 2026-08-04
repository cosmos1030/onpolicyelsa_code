"""vLLM subprocess server for FSDP+OPKD (TRL-style isolation).

vLLM runs in a completely separate subprocess with its own CUDA context
and torch.distributed state. Communication is via multiprocessing Queue.
This avoids ALL NCCL/dist conflicts between vLLM and the training process group.

Usage:
    from elsa.lib.vllm_proc import VLLMSubprocAdapter, launch_vllm_subprocess

    adapter = launch_vllm_subprocess(
        model_path, cuda_device_str, gpu_mem, max_len, enforce_eager,
        max_new_tokens, temperature)
    # Use adapter.generate(inputs, params) — same interface as vLLM LLM.generate()
    # Use adapter.sync_weights(state_dict) — update vLLM weights from training model
    adapter.shutdown()
"""
import os
import logging


# ─── Weight sync helper ──────────────────────────────────────────────────────

def _sync_weights_to_engine(engine, state_dict):
    """Apply state_dict to the vLLM engine's model in-place via load_weights().

    Works for TP=1 (single-GPU) via driver_worker. state_dict values must be
    CPU tensors (copied from FSDP summon_full_params with offload_to_cpu=True).
    """
    try:
        # vLLM 0.10+: model_executor lives under engine_core
        executor = (engine.llm_engine.engine_core.model_executor
                    if hasattr(engine.llm_engine, 'engine_core')
                    else engine.llm_engine.model_executor)
        model = executor.driver_worker.model_runner.model
        model.load_weights(state_dict.items())
    except Exception as e:
        logging.warning(f"[vllm_proc] sync_weights failed: {e}")


# ─── Subprocess entry-point ─────────────────────────────────────────────────

def _vllm_server_fn(model_path, cuda_device_str, gpu_mem, max_len, enforce_eager,
                    req_queue, resp_queue, log_path):
    """Runs inside the vLLM subprocess. Never call directly from training process."""
    import sys as _sys
    if log_path:
        _log_fh = open(log_path, 'w', buffering=1)
        _sys.stdout = _log_fh
        _sys.stderr = _log_fh
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device_str
    for _k in ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK',
               'GROUP_RANK', 'ROLE_RANK', 'TORCHELASTIC_RESTART_COUNT']:
        os.environ.pop(_k, None)
    os.environ['VLLM_USE_V1'] = '0'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['VLLM_HOST_IP'] = '127.0.0.1'
    import pathlib
    _tc = pathlib.Path.home() / '.triton_cache_vllm'
    _tc.mkdir(parents=True, exist_ok=True)
    os.environ['TRITON_CACHE_DIR'] = str(_tc)

    # Pre-initialize torch.distributed with a file store (avoids TCP port race/block).
    # vLLM skips its own TCPStore rendezvous when dist is already initialized.
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

    from vllm import LLM, SamplingParams

    engine = LLM(
        model_path,
        dtype='bfloat16',
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
        max_model_len=max_len,
        enforce_eager=enforce_eager,
    )
    resp_queue.put('ready')

    while True:
        req = req_queue.get()
        if req is None:
            resp_queue.put('done')
            break
        if isinstance(req, tuple) and req[0] == 'sync_weights':
            _, state_dict = req
            _sync_weights_to_engine(engine, state_dict)
            resp_queue.put('synced')
            continue
        prompt_ids_list, max_new, temp = req
        params = SamplingParams(max_tokens=max_new, temperature=temp, top_p=0.95)
        inputs = [{'prompt_token_ids': ids} for ids in prompt_ids_list]
        outputs = engine.generate(inputs, params)
        resp_queue.put([out.outputs[0].token_ids for out in outputs])


# ─── Fake output objects (match vLLM's CompletionOutput interface) ───────────

class _FakeOutputItem:
    __slots__ = ('token_ids',)

    def __init__(self, token_ids):
        self.token_ids = token_ids


class _FakeRequestOutput:
    __slots__ = ('outputs',)

    def __init__(self, token_ids):
        self.outputs = [_FakeOutputItem(token_ids)]


# ─── Adapter (same interface as vLLM LLM) ────────────────────────────────────

class VLLMSubprocAdapter:
    """Wraps the vLLM subprocess; exposes .generate() and .sync_weights().

    Inputs: list of TokensPrompt dicts (with key 'prompt_token_ids').
    Params: SamplingParams (reads .max_tokens and .temperature).
    Returns: list of _FakeRequestOutput with .outputs[0].token_ids.
    """

    def __init__(self, req_queue, resp_queue, proc, default_max_new, default_temp):
        self._req_q = req_queue
        self._resp_q = resp_queue
        self._proc = proc
        self._default_max_new = default_max_new
        self._default_temp = default_temp

    def _get_with_alive_check(self, poll_secs=30):
        """Block on resp_queue but raise RuntimeError if subprocess dies."""
        import queue as _queue
        while True:
            try:
                return self._resp_q.get(timeout=poll_secs)
            except _queue.Empty:
                if not self._proc.is_alive():
                    raise RuntimeError(
                        f"[VLLMSubprocAdapter] subprocess died (exitcode={self._proc.exitcode}) "
                        f"while waiting for response")
                logging.warning("[VLLMSubprocAdapter] still waiting for subprocess response…")

    def generate(self, inputs, params=None):
        max_new = params.max_tokens if params is not None else self._default_max_new
        temp = params.temperature if params is not None else self._default_temp
        prompt_ids_list = [inp['prompt_token_ids'] for inp in inputs]
        self._req_q.put((prompt_ids_list, max_new, temp))
        token_id_lists = self._get_with_alive_check()
        return [_FakeRequestOutput(tids) for tids in token_id_lists]

    def sync_weights(self, state_dict):
        """Send CPU state_dict to vLLM subprocess and block until applied."""
        self._req_q.put(('sync_weights', state_dict))
        result = self._get_with_alive_check()
        if result != 'synced':
            logging.warning(f"[VLLMSubprocAdapter] unexpected sync_weights response: {result!r}")

    def shutdown(self):
        try:
            self._req_q.put(None)
            self._resp_q.get(timeout=30)
        except Exception:
            pass
        self._proc.terminate()
        self._proc.join(timeout=10)


# ─── Launch helper ────────────────────────────────────────────────────────────

def launch_vllm_subprocess(model_path, cuda_device_str, gpu_mem, max_len,
                            enforce_eager, default_max_new, default_temp,
                            startup_timeout=300):
    """Spawn vLLM in a separate process; return a VLLMSubprocAdapter.

    Uses 'spawn' start method to ensure clean CUDA state in the child.
    Waits up to startup_timeout seconds for vLLM to be ready.
    """
    import multiprocessing as _mp
    ctx = _mp.get_context('spawn')
    req_q = ctx.Queue()
    resp_q = ctx.Queue()
    import tempfile as _tmp, pathlib as _pl
    _log_path = str(_pl.Path(_tmp.gettempdir()) / f'vllm_subproc_{os.getpid()}.log')
    logging.info(f"[vllm_proc] subprocess log: {_log_path}")
    proc = ctx.Process(
        target=_vllm_server_fn,
        args=(model_path, cuda_device_str, gpu_mem, max_len, enforce_eager,
              req_q, resp_q, _log_path),
        daemon=True,
    )
    # 'spawn' re-executes this process's entry script (main.py) top-level code in
    # the child BEFORE _vllm_server_fn runs (Python's spawn bootstrap does
    # runpy.run_path(main_path) to reconstruct __main__ for unpickling). If that
    # top-level code touches CUDA (e.g. torch.cuda.device_count()) before
    # _vllm_server_fn gets a chance to restrict CUDA_VISIBLE_DEVICES, CUDA locks
    # onto the full device list and the vLLM engine ends up on GPU 0 — colliding
    # with training rank 0 instead of its intended dedicated GPU. Restricting
    # CUDA_VISIBLE_DEVICES in THIS (parent) process right before start() ensures
    # the child inherits the restricted view from the very first line it runs.
    _orig_cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device_str
    try:
        proc.start()
    finally:
        if _orig_cvd is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = _orig_cvd
    msg = resp_q.get(timeout=startup_timeout)
    if msg != 'ready':
        proc.terminate()
        raise RuntimeError(f"vLLM subprocess did not send 'ready'; got: {msg!r}")
    return VLLMSubprocAdapter(req_q, resp_q, proc, default_max_new, default_temp)
