"""PoC: joint Gauss-Newton PCG reconstruction over one MLP block (gate_proj +
up_proj + down_proj, spanning the SiLU nonlinearity), applied on top of an
already-pruned ALPS checkpoint with the mask held completely fixed.

ALPS's own PCG backsolve (see pcg_correct_gmp_checkpoint.py) treats every
Linear independently: min_W ||X W^T - X W_dense^T||^2 per layer, ignoring how
that layer's own reconstruction error propagates through the nonlinearity
into the next layer. This script instead defines the whole MLP block as one
function F_theta(X) = down_proj(SiLU(gate_proj(X)) * up_proj(X)) and jointly
refines the ALIVE (mask==1) weights of all three Linears to match the DENSE
model's block output on the SAME input X -- X is the current (fixed-mask)
ALPS model's own hidden state at that point (same convention as ALPS's own
sequential PCG correction), so the only variable changed vs. the existing
baseline is "joint MLP-block reconstruction" vs. "independent per-Linear
reconstruction", holding everything else fixed.

No Jacobian is ever formed. Gauss-Newton normal equations
    (J^T J + lambda I) Delta = -J^T r,   r = F_theta(X) - Y_dense
are solved via matrix-free PCG: each matvec A(v) = J^T(J v) + lambda v is
computed as one forward-mode JVP (Jv) followed by one reverse-mode VJP
(J^T applied to that), i.e. the standard Hessian-vector-product trick
applied to Gauss-Newton -- no matrix bigger than the block's own activations
is ever materialized.

Usage:
    python mlp_joint_gnpcg.py <dense_model_path> <alps_model_path> \\
        --data_path <calib.jsonl> --layer_idx 12 \\
        --pcg_iters 1,2,5,10,20 --gn_outer 1 --damping 1e-3 \\
        --nsamples 128 --heldout 64 --seqlen 2048
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from qwen3_alps import get_ot_fw, _make_catcher, _layer_fwd

DEV = torch.device('cuda:0')


def get_model(path):
    def skip(*a, **k):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype='auto', trust_remote_code=True)
    model.seqlen = 2048
    return model


@torch.no_grad()
def propagate_to_layer(model, dataloader, dev, nsamples, layer_idx):
    """Run `model`'s own forward through layers[0..layer_idx-1] sequentially
    (ALPS's own catcher/hook convention) and return the resulting hidden
    states -- the model's OWN input to layers[layer_idx], i.e. X for that
    layer's MLP once the attention sublayer of layer_idx itself also runs."""
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
    return inps, cache  # inps = X feeding into layers[layer_idx]


@torch.no_grad()
def capture_mlp_io(model, layer_idx, layer_input, cache, dev, chunk=8):
    """Run just layers[layer_idx]'s attention+input_layernorm sublayer to get
    the MLP's actual input X_mlp, and the MLP's own output Y (for baseline
    recon-error logging), by hooking the mlp submodule during one real
    forward pass of that decoder layer."""
    layer = model.model.layers[layer_idx].to(dev)
    captured_in, captured_out = [], []

    def hook(_module, inp, out):
        captured_in.append(inp[0].detach())
        captured_out.append(out.detach())

    h = layer.mlp.register_forward_hook(hook)
    n = layer_input.shape[0]
    for i in range(0, n, chunk):
        _layer_fwd(layer, layer_input[i:i + chunk], cache)
    h.remove()
    layer.cpu()
    torch.cuda.empty_cache()

    X = torch.cat(captured_in, dim=0).reshape(-1, captured_in[0].shape[-1])
    Y = torch.cat(captured_out, dim=0).reshape(-1, captured_out[0].shape[-1])
    return X, Y


# ── Functional MLP block + matrix-free Gauss-Newton PCG ─────────────────────

CHUNK_TOKENS = 8192  # cap per-matmul token count so [chunk, intermediate_size]
                      # activations (gate/up/act, fp32) fit on a 24GB card --
                      # the full 128*2048=262144-token batch OOM'd at ~21GB
                      # just for one such tensor.


def block_forward(params: dict, x: torch.Tensor) -> torch.Tensor:
    """params: {'gate': W_gate, 'up': W_up, 'down': W_down}, no bias
    (Qwen3MLP convention). x: [tokens, hidden]. Chunked over tokens --
    each token is processed independently by the MLP (no cross-token
    coupling), so this is exact, not an approximation."""
    if x.shape[0] <= CHUNK_TOKENS:
        gate = F.linear(x, params['gate'])
        up = F.linear(x, params['up'])
        act = F.silu(gate) * up
        return F.linear(act, params['down'])
    outs = []
    for i in range(0, x.shape[0], CHUNK_TOKENS):
        outs.append(block_forward(params, x[i:i + CHUNK_TOKENS]))
    return torch.cat(outs, dim=0)


def _chunk_bounds(n, chunk=CHUNK_TOKENS):
    return [(i, min(i + chunk, n)) for i in range(0, n, chunk)]


def chunked_JtJv(params, x, v):
    """J^T @ (J @ v), summed over token chunks. Recomputes each chunk's
    forward+jvp+vjp FRESH every call (no cross-PCG-iteration graph caching)
    so peak memory is bounded to ONE chunk's saved-for-backward tensors
    regardless of how many chunks or PCG iterations there are -- caching all
    chunks' vjp closures simultaneously (to avoid this recompute) would hold
    ~n_chunks x one-chunk's-graph in memory at once, which is what actually
    OOM'd the naive whole-batch version in the first place, just shifted
    from "one huge chunk" to "many chunks kept alive at once."
    """
    acc = None
    for i, j in _chunk_bounds(x.shape[0]):
        xc = x[i:j]
        _, Jv_c = torch.func.jvp(lambda p: block_forward(p, xc), (params,), (v,))
        _, vfn = torch.func.vjp(lambda p: block_forward(p, xc), params)
        (g,) = vfn(Jv_c)
        acc = g if acc is None else {k: acc[k] + g[k] for k in g}
    return acc


def chunked_Jtr(params, x, y_ref):
    """(J^T @ r, F_theta(x)) where r = F_theta(x) - y_ref, summed/concatenated
    over token chunks, same fresh-per-chunk-graph policy as chunked_JtJv."""
    acc = None
    outs = []
    for i, j in _chunk_bounds(x.shape[0]):
        xc = x[i:j]
        oc, vfn = torch.func.vjp(lambda p: block_forward(p, xc), params)
        rc = oc - y_ref[i:j]
        (g,) = vfn(rc)
        acc = g if acc is None else {k: acc[k] + g[k] for k in g}
        outs.append(oc.detach())
    return acc, torch.cat(outs, dim=0)


def _tree_dot(a: dict, b: dict) -> torch.Tensor:
    return sum((a[k] * b[k]).sum() for k in a)


def _tree_add(a: dict, b: dict, alpha=1.0) -> dict:
    return {k: a[k] + alpha * b[k] for k in a}


def _tree_scale(a: dict, alpha) -> dict:
    return {k: alpha * a[k] for k in a}


def _tree_zeros_like(a: dict) -> dict:
    return {k: torch.zeros_like(v) for k, v in a.items()}


def gn_pcg_solve(params, x, y_ref, masks, damping, maxiter):
    """Solve (J^T J + damping*I) Delta = -J^T r for Delta restricted to the
    alive (masks==1) entries of each param, via PCG. masks: dict matching
    params' shapes, 1.0/True at alive positions. Returns Delta (dict, zero
    at dead positions)."""

    def masked(d):
        return {k: d[k] * masks[k] for k in d}

    def A(v):
        # v is masked already; JtJv restricted to alive positions on the way
        # out too (dead positions must stay exactly 0).
        JtJv = chunked_JtJv(params, x, v)
        return _tree_add(masked(JtJv), v, alpha=damping)

    Jtr0, _ = chunked_Jtr(params, x, y_ref)
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


# ── Trajectory metrics ───────────────────────────────────────────────────────

@torch.no_grad()
def recon_error(params, x, y_ref) -> float:
    out = block_forward(params, x)
    return (out - y_ref).pow(2).sum(dim=-1).mean().sqrt().item()  # RMS per token


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
    p.add_argument('--save_prefix', default=None)
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

    print(f'Loading calib ({args.nsamples}) + held-out ({args.heldout}) data ...')
    calib_loader, _ = get_ot_fw(args.nsamples, args.seed, args.seqlen, tokenizer, args.data_path)
    heldout_loader, _ = get_ot_fw(args.heldout, args.seed + 10_000, args.seqlen, tokenizer, args.data_path)

    print(f'Propagating to layer {args.layer_idx} (ALPS model\'s own trajectory) ...')
    alps_model.seqlen = args.seqlen
    calib_in, cache = propagate_to_layer(alps_model, calib_loader, DEV, args.nsamples, args.layer_idx)
    heldout_in, cache_h = propagate_to_layer(alps_model, heldout_loader, DEV, args.heldout, args.layer_idx)

    print('Capturing MLP block input/output (calib + heldout) ...')
    X_calib, Y_alps_calib = capture_mlp_io(alps_model, args.layer_idx, calib_in, cache, DEV)
    X_heldout, Y_alps_heldout = capture_mlp_io(alps_model, args.layer_idx, heldout_in, cache_h, DEV)

    mlp = alps_model.model.layers[args.layer_idx].mlp
    dense_mlp = dense_model.model.layers[args.layer_idx].mlp

    params = {
        'gate': mlp.gate_proj.weight.detach().clone().float().to(DEV).requires_grad_(False),
        'up': mlp.up_proj.weight.detach().clone().float().to(DEV).requires_grad_(False),
        'down': mlp.down_proj.weight.detach().clone().float().to(DEV).requires_grad_(False),
    }
    masks = {k: (v != 0).float() for k, v in params.items()}
    for k, v in masks.items():
        n_alive = v.sum().item()
        print(f'  {k}: {tuple(v.shape)}, alive={int(n_alive)}/{v.numel()} ({n_alive/v.numel()*100:.1f}%)')

    dense_params = {
        'gate': dense_mlp.gate_proj.weight.detach().clone().float().to(DEV),
        'up': dense_mlp.up_proj.weight.detach().clone().float().to(DEV),
        'down': dense_mlp.down_proj.weight.detach().clone().float().to(DEV),
    }
    X_calib = X_calib.float()
    X_heldout = X_heldout.float()
    with torch.no_grad():
        Y_ref_calib = block_forward(dense_params, X_calib)
        Y_ref_heldout = block_forward(dense_params, X_heldout)

    e0_calib = recon_error(params, X_calib, Y_ref_calib)
    e0_heldout = recon_error(params, X_heldout, Y_ref_heldout)
    print(f'[baseline: ALPS layer-wise PCG only] calib_rms={e0_calib:.5f} heldout_rms={e0_heldout:.5f}')

    pcg_iters = [int(x) for x in args.pcg_iters.split(',')]
    results = [{'pcg_iter': 0, 'gn_outer': 0, 'calib_rms': e0_calib, 'heldout_rms': e0_heldout}]

    for target_iters in pcg_iters:
        cur_params = {k: v.clone() for k, v in params.items()}
        t0 = time.time()
        done_iters = 0
        for outer in range(args.gn_outer):
            remaining = target_iters - done_iters if outer == args.gn_outer - 1 else target_iters // args.gn_outer
            remaining = max(remaining, 1)
            delta = gn_pcg_solve(cur_params, X_calib, Y_ref_calib, masks, args.damping, remaining)
            cur_params = _tree_add(cur_params, delta, alpha=1.0)
            done_iters += remaining
            e_calib = recon_error(cur_params, X_calib, Y_ref_calib)
            e_held = recon_error(cur_params, X_heldout, Y_ref_heldout)
            print(f'  pcg_iters={target_iters} gn_outer={outer+1}/{args.gn_outer} '
                  f'({time.time()-t0:.1f}s): calib_rms={e_calib:.5f} heldout_rms={e_held:.5f}')
            results.append({'pcg_iter': target_iters, 'gn_outer': outer + 1,
                             'calib_rms': e_calib, 'heldout_rms': e_held})

        if args.save_prefix:
            out_dtype = mlp.gate_proj.weight.dtype
            torch.save(
                {k: v.to(out_dtype).cpu() for k, v in cur_params.items()},
                f'{args.save_prefix}_layer{args.layer_idx}_iters{target_iters}.pt',
            )

    print('\n=== trajectory summary ===')
    for r in results:
        print(r)


if __name__ == '__main__':
    main()
