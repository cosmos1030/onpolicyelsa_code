"""vLLM sidecar server client for FSDP+OPKD.

vLLM runs in lib/vllm_server_standalone.py, launched via subprocess.Popen (a
fresh `python` exec, NOT multiprocessing.Process) so it is a genuinely
independent OS process -- no inherited CUDA context, no inherited
torch.distributed/NCCL state, no torchrun env vars. This replaced an earlier
multiprocessing.Process-based version that hit three bugs in a row once
tensor_parallel_size>1 was added: a daemonic-process-can't-have-children
assertion, a fork-vs-spawn CUDA re-init crash, and (the one that killed the
approach) a silent deadlock in vLLM's own inter-worker NCCL/gloo rendezvous --
all symptoms of nesting vLLM's distributed runtime inside torchrun's process
tree instead of giving it a clean one. See vllm's own multiprocessing design
doc and its `external_launcher` executor backend, which exists precisely
because torchrun-owned ranks and vLLM-owned ranks don't mix well when forced
into the same process.

Communication is over a Unix-domain-socket multiprocessing.connection --
same wire protocol (plain send/recv of pickled Python objects) the old
Queue-based version used, just not tied to a parent/child relationship.

Usage:
    from elsa.lib.vllm_proc import launch_vllm_server

    adapter = launch_vllm_server(
        model_path, cuda_device_str, gpu_mem, max_len, enforce_eager,
        max_new_tokens, temperature, tensor_parallel_size=2)
    # Use adapter.generate(inputs, params) — same interface as vLLM LLM.generate()
    # Use adapter.sync_weights(state_dict) — update vLLM weights from training model
    adapter.shutdown()
"""
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import uuid
from multiprocessing.connection import Client

# torch monkey-patches multiprocessing's ForkingPickler so ANY torch.Tensor
# sent over a plain multiprocessing.connection (CPU tensors included, e.g. a
# sync_weights() state_dict) uses its own reduce_storage. That reducer's
# default 'file_descriptor' strategy needs the receiver to open a callback
# connection to the sender's multiprocessing.resource_sharer, authenticated
# with multiprocessing.current_process().authkey -- which two independently
# subprocess.Popen'd processes never share (that key is only ever inherited
# automatically across an actual multiprocessing.Process parent/child, which
# is exactly what this module deliberately does NOT use -- see module
# docstring). Symptom: AuthenticationError: digest sent was rejected, raised
# from inside conn.recv() the first time a tensor-bearing payload arrives.
# 'file_system' shares tensors via named /dev/shm segments instead, which the
# receiver opens directly by name -- no cross-process authkey handshake.
import torch.multiprocessing as _torch_mp
_torch_mp.set_sharing_strategy('file_system')


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

class VLLMServerAdapter:
    """Wraps the standalone vLLM server process; exposes .generate() and
    .sync_weights() with the same call signatures the old in-process
    multiprocessing adapter used, so gmp_trainer.py's call sites don't care
    which transport is underneath.

    Inputs: list of TokensPrompt dicts (with key 'prompt_token_ids').
    Params: SamplingParams (reads .max_tokens and .temperature).
    Returns: list of _FakeRequestOutput with .outputs[0].token_ids.
    """

    def __init__(self, conn, proc, default_max_new, default_temp, sleep_mode=False):
        self._conn = conn
        self._proc = proc
        self._default_max_new = default_max_new
        self._default_temp = default_temp
        if sleep_mode:
            self.sleep = self._sleep_impl
            self.wake_up = self._wake_impl

    def _recv_with_alive_check(self, poll_secs=30):
        import multiprocessing.connection as _mpc
        while True:
            if _mpc.wait([self._conn], timeout=poll_secs):
                return self._conn.recv()
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"[VLLMServerAdapter] server process died (exitcode={self._proc.returncode}) "
                    f"while waiting for response")
            logging.warning("[VLLMServerAdapter] still waiting for server response…")

    def generate(self, inputs, params=None):
        max_new = params.max_tokens if params is not None else self._default_max_new
        temp = params.temperature if params is not None else self._default_temp
        prompt_ids_list = [inp['prompt_token_ids'] for inp in inputs]
        self._conn.send((prompt_ids_list, max_new, temp))
        token_id_lists = self._recv_with_alive_check()
        return [_FakeRequestOutput(tids) for tids in token_id_lists]

    def sync_weights(self, state_dict):
        """Send CPU state_dict to the vLLM server and block until applied."""
        self._conn.send(('sync_weights', state_dict))
        result = self._recv_with_alive_check()
        if result != 'synced':
            logging.warning(f"[VLLMServerAdapter] unexpected sync_weights response: {result!r}")

    # _sleep_impl/_wake_impl are bound to the public names `sleep`/`wake_up` in
    # __init__ ONLY when the server was launched with --sleep-mode.
    # gmp_trainer.py's _opkd_vllm_sleep/_opkd_vllm_wake gate on
    # hasattr(engine, 'sleep'), so leaving them unbound makes an adapter
    # correctly report "no sleep support" and degrade to never sleeping --
    # the right behavior for the FSDP layout, where the sidecar owns a
    # dedicated GPU and releasing its memory between rollouts buys nothing.
    # Defining them unconditionally on the class would make that hasattr
    # always true and start sleeping the FSDP sidecar against a server that
    # cannot honor it.
    def _sleep_impl(self, level=1):
        """Offload weights to CPU + drop KV cache, releasing this sidecar's GPU
        memory back to the trainer sharing the same device."""
        self._conn.send(('sleep', level))
        result = self._recv_with_alive_check()
        if result != 'ok':
            raise RuntimeError(f"[VLLMServerAdapter] sleep failed: {result!r}")

    def _wake_impl(self):
        """Re-map weights/KV cache before a rollout batch."""
        self._conn.send(('wake',))
        result = self._recv_with_alive_check()
        if result != 'ok':
            raise RuntimeError(f"[VLLMServerAdapter] wake_up failed: {result!r}")

    def shutdown(self):
        try:
            self._conn.send(None)
            self._conn.recv()
        except Exception:
            pass
        self._conn.close()
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()


# ─── Launch helper ────────────────────────────────────────────────────────────

_ELASTIC_LAUNCH_ENV_KEYS = (
    'MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE', 'LOCAL_RANK',
    'GROUP_RANK', 'ROLE_RANK', 'TORCHELASTIC_RESTART_COUNT',
    'TORCHELASTIC_RUN_ID', 'LOCAL_WORLD_SIZE',
)


def launch_vllm_server(model_path, cuda_device_str, gpu_mem, max_len,
                        enforce_eager, default_max_new, default_temp,
                        startup_timeout=300, tensor_parallel_size=1,
                        sleep_mode=False):
    """Launch vLLM as an independent OS process; return a VLLMServerAdapter.

    Waits up to startup_timeout seconds for the server's readiness marker.

    sleep_mode=True builds the engine with vLLM's sleep support and exposes
    .sleep()/.wake_up() on the returned adapter -- needed when this sidecar
    SHARES a GPU with the trainer, so its footprint is not permanently
    resident. Leave False when it has a dedicated GPU (the FSDP layout).
    """
    addr = str(pathlib.Path(tempfile.gettempdir()) / f'vllm_server_{uuid.uuid4().hex}.sock')
    authkey = uuid.uuid4().hex
    log_path = str(pathlib.Path(tempfile.gettempdir()) / f'vllm_server_{os.getpid()}.log')
    script = str(pathlib.Path(__file__).parent / 'vllm_server_standalone.py')
    logging.info(f"[vllm_proc] launching standalone vLLM server on GPU(s) {cuda_device_str} "
                 f"(tp_size={tensor_parallel_size}): log={log_path}")

    # A clean env (no torchrun rank/rendezvous vars) so vLLM's own distributed
    # runtime never sees state it would try to (mis)interpret as its own.
    clean_env = {k: v for k, v in os.environ.items() if k not in _ELASTIC_LAUNCH_ENV_KEYS}

    cmd = [sys.executable, script,
           '--model', model_path,
           '--cuda-devices', cuda_device_str,
           '--tp-size', str(tensor_parallel_size),
           '--gpu-mem', str(gpu_mem),
           '--max-len', str(max_len),
           '--address', addr,
           '--authkey', authkey]
    if enforce_eager:
        cmd.append('--enforce-eager')
    if sleep_mode:
        cmd.append('--sleep-mode')

    log_fh = open(log_path, 'w')
    proc = subprocess.Popen(cmd, env=clean_env, stdout=log_fh, stderr=subprocess.STDOUT)

    deadline = time.time() + startup_timeout
    ready = False
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM server process exited early (code={proc.returncode}); see {log_path}")
        with open(log_path) as f:
            if 'VLLM_SERVER_READY' in f.read():
                ready = True
                break
        time.sleep(1)
    if not ready:
        proc.terminate()
        raise RuntimeError(f"vLLM server did not become ready within {startup_timeout}s; see {log_path}")

    conn = None
    conn_deadline = time.time() + 30
    while time.time() < conn_deadline:
        try:
            conn = Client(addr, family='AF_UNIX', authkey=authkey.encode())
            break
        except (ConnectionRefusedError, FileNotFoundError):
            time.sleep(1)
    if conn is None:
        proc.terminate()
        raise RuntimeError(f"Could not connect to vLLM server socket at {addr}; see {log_path}")

    logging.info("[vllm_proc] standalone vLLM server ready")
    return VLLMServerAdapter(conn, proc, default_max_new, default_temp, sleep_mode=sleep_mode)
