"""Scope-widening experiment (step 2 of the incremental sweep): same joint
Gauss-Newton PCG idea as mlp_joint_gnpcg.py, but the "block" is now a WHOLE
decoder layer (self_attn + MLP, spanning RoPE/softmax attention AND SiLU)
instead of just the MLP. Mask stays fixed on q/k/v/o_proj + gate/up/down_proj
(7 Linears); RMSNorm weights and rotary buffers are left untouched (never
pruned by ALPS, and not meaningful to update via this mask-fixed PCG anyway).

Purpose: find where Gauss-Newton's local-linearization assumption starts to
break down as the reconstruction scope widens (MLP-only -> MLP+attention ->
[not yet implemented] multiple adjacent layers) -- tracked via the SAME
calib/held-out recon trajectory used for the MLP-only PoC, directly
comparable since it's the identical checkpoint/data/damping/pcg_iters grid.

IMPORTANT difference from the MLP-only version: MLP processes each token
independently, so mlp_joint_gnpcg.py could freely chunk over a flattened
[tokens, hidden] tensor. Self-attention mixes information ACROSS tokens
within one sequence (causal attention), so chunking here must be done per
SEQUENCE (a full [1, seqlen, hidden] sample), never by splitting a sequence
mid-way -- otherwise attention would silently see a truncated context and
produce wrong (not just approximate) outputs.

Reuses transformers' actual Qwen3DecoderLayer.forward via
torch.func.functional_call (only overriding the 7 Linear .weight tensors)
instead of hand-reimplementing attention math -- avoids subtly getting RoPE/
causal-masking wrong.

Usage:
    python layer_joint_gnpcg.py <dense_model_path> <alps_model_path> \\
        --data_path <calib.jsonl> --layer_idx 12 \\
        --pcg_iters 1,2,5,10,20 --gn_outer 1 --damping 1e-3 \\
        --nsamples 128 --heldout 64 --seqlen 2048
"""
import argparse
import gc
import random
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from qwen3_alps import get_ot_fw, _make_catcher, _layer_fwd
from mlp_joint_gnpcg import get_model, _tree_add, _tree_scale, _tree_dot, _tree_zeros_like

DEV = torch.device('cuda:0')

LINEAR_NAMES = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
                 'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']

SEQ_CHUNK = 4  # sequences per matvec chunk -- MATH-backend SDPA (forced,
               # see layer_forward) materializes the full [batch, heads,
               # seqlen, seqlen] attention score matrix explicitly (unlike
               # fused kernels), and jvp+vjp both need it -- OOM'd a 24GB
               # card even at chunk=1 for a 2048-token sequence; on an
               # A100-80GB there's enough headroom to batch a few sequences.


def layer_forward(params: dict, x: torch.Tensor, layer_module, layer_kwargs: dict) -> torch.Tensor:
    """params: {short_name: weight_tensor} for the 7 Linears above. x:
    [n_seq, seqlen, hidden]. Runs the REAL decoder layer forward with those
    weights substituted in via functional_call (everything else -- norms,
    rotary buffers -- comes from the module's own current state, unchanged)."""
    # functional_call expects names relative to the module passed in -- since
    # we pass the single decoder layer module directly (not indexed through a
    # parent ModuleList), params' keys (e.g. 'self_attn.q_proj.weight') are
    # already correct as-is.
    #
    # Forced to the MATH (naive matmul+softmax+matmul) SDPA backend: the
    # fused kernels (flash / mem-efficient) don't implement forward-mode AD
    # ("Trying to use forward AD with _scaled_dot_product_efficient_attention
    # that does not support it"), which torch.func.jvp needs. MATH is pure
    # composable ops, so both forward- and reverse-mode AD work through it.
    with sdpa_kernel(SDPBackend.MATH):
        out = torch.func.functional_call(layer_module, params, (x,), layer_kwargs)
    return out[0] if isinstance(out, tuple) else out


def _seq_chunk_bounds(n, chunk):
    return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]


def _release(*objs):
    """torch.func.vjp's returned closure holds a reference to the whole
    per-chunk autograd graph (all saved-for-backward activations). Autograd
    graphs commonly contain reference cycles, so CPython's refcounting alone
    does NOT free them the moment they go out of scope -- they sit until the
    generational GC happens to run, which on a tight per-chunk loop through
    an entire decoder layer's attention (O(seqlen^2) activations each) is
    exactly how this ran a 24GB card AND an 80GB card out of memory even at
    a single-sequence chunk size. Explicitly deleting and forcing collection
    every chunk is the fix."""
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def chunked_JtJv(params, x, v, layer_module, layer_kwargs, chunk=SEQ_CHUNK):
    acc = None
    for i, j in _seq_chunk_bounds(x.shape[0], chunk):
        xc = x[i:j]
        _, Jv_c = torch.func.jvp(lambda p: layer_forward(p, xc, layer_module, layer_kwargs), (params,), (v,))
        _, vfn = torch.func.vjp(lambda p: layer_forward(p, xc, layer_module, layer_kwargs), params)
        (g,) = vfn(Jv_c)
        g = {k: g[k].detach().clone() for k in g}
        acc = g if acc is None else {k: acc[k] + g[k] for k in g}
        _release(Jv_c, vfn)
    return acc


def chunked_Jtr(params, x, y_ref, layer_module, layer_kwargs, chunk=SEQ_CHUNK):
    acc = None
    outs = []
    for i, j in _seq_chunk_bounds(x.shape[0], chunk):
        xc = x[i:j]
        oc, vfn = torch.func.vjp(lambda p: layer_forward(p, xc, layer_module, layer_kwargs), params)
        rc = oc - y_ref[i:j]
        (g,) = vfn(rc)
        g = {k: g[k].detach().clone() for k in g}
        acc = g if acc is None else {k: acc[k] + g[k] for k in g}
        outs.append(oc.detach().clone())
        _release(oc, vfn, rc)
    return acc, torch.cat(outs, dim=0)


def gn_pcg_solve(params, x, y_ref, masks, damping, maxiter, layer_module, layer_kwargs, chunk=SEQ_CHUNK):
    def masked(d):
        return {k: d[k] * masks[k] for k in d}

    def A(v):
        JtJv = chunked_JtJv(params, x, v, layer_module, layer_kwargs, chunk)
        return _tree_add(masked(JtJv), v, alpha=damping)

    Jtr0, _ = chunked_Jtr(params, x, y_ref, layer_module, layer_kwargs, chunk)
    b = masked(_tree_scale(Jtr0, -1.0))

    delta = _tree_zeros_like(params)
    r = {k: b[k].clone() for k in b}
    p = {k: r[k].clone() for k in r}
    rs_old = _tree_dot(r, r)

    for it in range(maxiter):
        Ap = A(p)
        pAp = _tree_dot(p, Ap)
        if pAp.item() == 0:
            break
        alpha = rs_old / pAp
        delta = _tree_add(delta, p, alpha=alpha.item())
        r = _tree_add(r, Ap, alpha=-alpha.item())
        rs_new = _tree_dot(r, r)
        if rs_new.item() < 1e-12:
            break
        beta = rs_new / rs_old
        p = _tree_add(r, p, alpha=beta.item())
        rs_old = rs_new

    return delta


@torch.no_grad()
def recon_error(params, x, y_ref, layer_module, layer_kwargs) -> float:
    out = layer_forward(params, x, layer_module, layer_kwargs)
    return (out - y_ref).pow(2).sum(dim=-1).mean().sqrt().item()


@torch.no_grad()
def propagate_to_layer(model, dataloader, dev, nsamples, layer_idx):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    dtype = next(iter(model.parameters())).dtype
    hidden = model.config.hidden_size
    inps = torch.zeros((nsamples, model.seqlen, hidden), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    Catcher = _make_catcher(inps, cache, nsamples)
    layers[0] = Catcher(layers[0])
    for inp, _ in dataloader:
        try:
            model(inp.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    for i in range(layer_idx):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = _layer_fwd(layer, inps[j].unsqueeze(0), cache)
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return inps, cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dense_model_path')
    p.add_argument('alps_model_path')
    p.add_argument('--data_path', required=True)
    p.add_argument('--layer_idx', type=int, required=True)
    p.add_argument('--pcg_iters', default='1,2,5,10,20')
    p.add_argument('--gn_outer', type=int, default=1)
    p.add_argument('--damping', type=float, default=1e-3)
    p.add_argument('--nsamples', type=int, default=128)
    p.add_argument('--heldout', type=int, default=64)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seq_chunk', type=int, default=SEQ_CHUNK,
                    help='sequences per PCG matvec chunk -- lower if OOM (MATH-backend attention memory ~ seq_chunk * seqlen^2)')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.alps_model_path, use_fast=False, trust_remote_code=True)

    print(f'Loading dense model from {args.dense_model_path} ...')
    dense_model = get_model(args.dense_model_path).eval()
    print(f'Loading ALPS model from {args.alps_model_path} ...')
    alps_model = get_model(args.alps_model_path).eval()
    alps_model.seqlen = args.seqlen

    print(f'Loading calib ({args.nsamples}) + held-out ({args.heldout}) data ...')
    calib_loader, _ = get_ot_fw(args.nsamples, args.seed, args.seqlen, tokenizer, args.data_path)
    heldout_loader, _ = get_ot_fw(args.heldout, args.seed + 10_000, args.seqlen, tokenizer, args.data_path)

    print(f'Propagating to layer {args.layer_idx} ...')
    calib_in, cache = propagate_to_layer(alps_model, calib_loader, DEV, args.nsamples, args.layer_idx)
    heldout_in, cache_h = propagate_to_layer(alps_model, heldout_loader, DEV, args.heldout, args.layer_idx)

    layer_kwargs = {k: cache[k] for k in ('attention_mask', 'position_ids', 'position_embeddings') if cache.get(k) is not None}
    layer_kwargs_h = {k: cache_h[k] for k in ('attention_mask', 'position_ids', 'position_embeddings') if cache_h.get(k) is not None}

    layer = alps_model.model.layers[args.layer_idx].to(DEV)
    dense_layer = dense_model.model.layers[args.layer_idx].to(DEV)

    params = {n: layer.get_parameter(n).detach().clone().float() for n in LINEAR_NAMES}
    masks = {k: (v != 0).float() for k, v in params.items()}
    for k, v in masks.items():
        n_alive = v.sum().item()
        print(f'  {k}: {tuple(v.shape)}, alive={int(n_alive)}/{v.numel()} ({n_alive/v.numel()*100:.1f}%)')

    dense_params = {n: dense_layer.get_parameter(n).detach().clone().float() for n in LINEAR_NAMES}

    with torch.no_grad():
        Y_ref_calib = layer_forward(dense_params, calib_in.float(), dense_layer, layer_kwargs)
        Y_ref_heldout = layer_forward(dense_params, heldout_in.float(), dense_layer, layer_kwargs_h)

    calib_in_f = calib_in.float()
    heldout_in_f = heldout_in.float()

    e0_calib = recon_error(params, calib_in_f, Y_ref_calib, layer, layer_kwargs)
    e0_heldout = recon_error(params, heldout_in_f, Y_ref_heldout, layer, layer_kwargs_h)
    print(f'[baseline: ALPS layer-wise PCG only] calib_rms={e0_calib:.5f} heldout_rms={e0_heldout:.5f}')

    pcg_iters = [int(x) for x in args.pcg_iters.split(',')]
    for target_iters in pcg_iters:
        cur_params = {k: v.clone() for k, v in params.items()}
        t0 = time.time()
        done_iters = 0
        for outer in range(args.gn_outer):
            remaining = target_iters - done_iters if outer == args.gn_outer - 1 else target_iters // args.gn_outer
            remaining = max(remaining, 1)
            delta = gn_pcg_solve(cur_params, calib_in_f, Y_ref_calib, masks, args.damping, remaining, layer, layer_kwargs, chunk=args.seq_chunk)
            cur_params = _tree_add(cur_params, delta, alpha=1.0)
            done_iters += remaining
            e_calib = recon_error(cur_params, calib_in_f, Y_ref_calib, layer, layer_kwargs)
            e_held = recon_error(cur_params, heldout_in_f, Y_ref_heldout, layer, layer_kwargs_h)
            print(f'  pcg_iters={target_iters} gn_outer={outer+1}/{args.gn_outer} '
                  f'({time.time()-t0:.1f}s): calib_rms={e_calib:.5f} heldout_rms={e_held:.5f}')


if __name__ == '__main__':
    main()
