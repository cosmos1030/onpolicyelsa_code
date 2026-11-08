"""Iterative one-shot pruning inside the GMP training loop.

Rebuttal baseline (SWVr (ii)/(iv)): instead of SCOUT's saliency + trust-region
mask updates, prune the CURRENT student with an established one-shot method
(SparseGPT or ALPS) at a few fixed steps, raising sparsity along a schedule,
and train with the usual objective (NTP+KD+OPD) between stages. Run with
--gmp_fixed_mask=true from a dense start: the mask is re-read from the weight
zeros after every stage and frozen in between.

Each stage is a standard layer-sequential one-shot pass over the student as it
is at that step (reconstruction target = the current layer's own outputs on
the calibration windows, the usual SparseGPT/ALPS setup), with uniform
per-layer sparsity, as both methods prune.  Calibration: nsamples x seqlen
windows from the fixed OT80/FW20 corpus, the same sampler (ALPS get_ot_fw) and
size (128 x 2048) as the paper's one-shot baselines.  The windows are drawn
once and reused at every stage.

Both upstream modules flip torch's global TF32 switches off at import; that
must not leak into training, so the switches are saved and restored around
the import and around every pruning pass.
"""
import logging
import math
import os
import sys
import time
import types
from contextlib import contextmanager

import torch
import torch.nn as nn

_ALPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ALPS'))


@contextmanager
def _tf32(enabled=None):
    """Save torch's TF32 switches, optionally set them, restore on exit."""
    m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        if enabled is not None:
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.backends.cudnn.allow_tf32 = enabled
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = m
        torch.backends.cudnn.allow_tf32 = c


def _import_alps():
    with _tf32():                                 # alps.py sets TF32 off at import
        if _ALPS_DIR not in sys.path:
            sys.path.append(_ALPS_DIR)            # append: never shadow elsa modules
        import qwen3_alps                         # noqa: E402
    return qwen3_alps


class _SparseGPT:
    """SparseGPT (Frantar & Alistarh, 2023), unstructured part of the reference
    implementation (IST-DASLab/sparsegpt sparsegpt.py), unchanged in the math:
    Hessian H = 2/n sum x x^T, damp 1%, blocksize 128, OBS mask selection by
    w^2 / [H^-1]_ii^2 within each block with error feedback."""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.rows, self.columns = layer.weight.shape
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        inp = inp.reshape(-1, inp.shape[-1]).t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(self, sparsity, blocksize=128, percdamp=.01):
        W = self.layer.weight.data.clone().float()
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        Losses = torch.zeros(self.rows, device=self.dev)
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
            thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
            mask1 = tmp <= thresh
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()
                q[mask1[:, i]] = 0
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        self.layer.weight.data.copy_(W.reshape(self.layer.weight.shape).to(self.layer.weight.dtype))
        return Losses.sum().item()

    def free(self):
        self.H = None


def cubic_sparsities(final_sparsity, n_stages):
    """s_k = s_f * (1 - (1 - k/K)^3), k = 1..K -- the cubic ramp of Zhu & Gupta
    (2017) sampled at the K stage boundaries; s_K = s_f."""
    return [final_sparsity * (1.0 - (1.0 - k / n_stages) ** 3) for k in range(1, n_stages + 1)]


class OneShotStager:
    """Holds the stage plan and the calibration windows; run() prunes one stage."""

    def __init__(self, method, stage_steps, stage_sparsities, tokenizer, data_path,
                 nsamples=128, seqlen=2048, seed=42, alps_rho=300.0):
        if method not in ('sparsegpt', 'alps'):
            raise ValueError(f"unknown one-shot method {method!r}")
        if len(stage_steps) != len(stage_sparsities):
            raise ValueError("stage_steps and stage_sparsities differ in length")
        if any(b <= a for a, b in zip(stage_sparsities, stage_sparsities[1:])):
            raise ValueError(f"stage sparsities must increase: {stage_sparsities}")
        self.method = method
        self.plan = dict(zip(stage_steps, stage_sparsities))
        self.tokenizer, self.data_path = tokenizer, data_path
        self.nsamples, self.seqlen, self.seed, self.alps_rho = nsamples, seqlen, seed, alps_rho
        self._loader = None
        self._q = _import_alps()

    def _calib(self):
        if self._loader is None:
            t0 = time.time()
            self._loader, _ = self._q.get_ot_fw(self.nsamples, self.seed, self.seqlen,
                                                self.tokenizer, self.data_path)
            logging.info(f"[oneshot] calibration: {len(self._loader)} x {self.seqlen} windows "
                         f"from {self.data_path} (seed {self.seed}) in {time.time() - t0:.0f}s")
        return self._loader

    def _sparsegpt_sequential(self, model, loader, dev, sp):
        """Layer-sequential SparseGPT over model.model.layers: the reference
        llama_sequential loop, with ALPS's Qwen3 catcher / layer-forward helpers
        so both methods see identical calibration inputs."""
        q = self._q
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        if hasattr(model.model, 'norm'):
            model.model.norm = model.model.norm.to(dev)
        q._rope_to(model, dev)
        layers[0] = layers[0].to(dev)
        dtype = next(iter(model.parameters())).dtype
        n = self.nsamples
        inps = torch.zeros((n, self.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
        cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}
        layers[0] = q._make_catcher(inps, cache, n)(layers[0])
        for batch in loader:
            try:
                model(batch[0].to(dev))
            except ValueError:
                pass
        layers[0] = layers[0].module
        outs = torch.zeros_like(inps)
        tot, nnz = 0, 0
        for i in range(len(layers)):
            layer = layers[i].to(dev)
            full = q.find_layers(layer)
            gpts = {name: _SparseGPT(full[name]) for name in full}

            def _hook(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data)
                return tmp
            handles = [full[name].register_forward_hook(_hook(name)) for name in full]
            for j in range(n):
                q._layer_fwd(layer, inps[j].unsqueeze(0), cache)
            for h in handles:
                h.remove()
            for name in full:
                gpts[name].fasterprune(sp)
                w = full[name].weight.data
                tot += w.numel()
                nnz += (w != 0).sum().item()
                gpts[name].free()
            for j in range(n):
                outs[j] = q._layer_fwd(layer, inps[j].unsqueeze(0), cache)
            del gpts
            torch.cuda.empty_cache()
            inps, outs = outs, inps
        return 1 - nnz / tot

    @torch.no_grad()
    def run(self, model, step, dev):
        """Prune `model` (unwrapped HF CausalLM) in place to this stage's
        sparsity. Returns (target, achieved, seconds). Leaves every module on
        `dev`; Parameter objects are never replaced, so the optimizer and the
        mask manager keep valid references."""
        sp = self.plan[step]
        loader = self._calib()
        t0 = time.time()
        was_training = model.training
        use_cache = model.config.use_cache
        model.eval()
        model.config.use_cache = False
        model.seqlen = self.seqlen
        try:
            with _tf32(False):
                if self.method == 'sparsegpt':
                    achieved = self._sparsegpt_sequential(model, loader, dev, sp)
                else:
                    args = types.SimpleNamespace(
                        sp=sp, nm_n=0, nm_m=0, rho=self.alps_rho, nsamples=self.nsamples,
                        corrected_target=False, dense_cache_dir=None, skip_layer_names={},
                        rollout_refine=False)
                    achieved = self._q.qwen3_sequential(model, loader, dev, args)
        finally:
            model.to(dev)                         # ALPS parks finished layers on CPU
            model.config.use_cache = use_cache
            model.train(was_training)
            torch.cuda.empty_cache()
        return sp, achieved, time.time() - t0
