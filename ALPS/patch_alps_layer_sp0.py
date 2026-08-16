"""
Re-run ALPS's own Hessian-collection + ADMM reconstruction math on ONE
decoder layer of an already-pruned checkpoint, but with sp=0.0 for that
layer -- i.e. everything ALPS normally does to a layer EXCEPT actually
removing any weights. Distinguishes "does the ADMM/CG reconstruction itself
perturb a layer's weights, independent of sparsity" from "does removing
information (real sparsity) hurt" -- a cheaper, more targeted variant of
patch_dense_layer.py's "swap back to literal original dense weights"
ablation (that one skips ALPS's math entirely; this one runs it with sp=0).

Cheap because ALPS processes layers strictly sequentially with no feedback
from later layers to earlier ones (see qwen3_alps.py's qwen3_sequential):
layers before the target layer are already correctly pruned in the existing
checkpoint (identical to what a full re-run would produce), so only the
target layer's own Hessian collection + ADMM need to be redone -- no need
to re-run the full multi-hour pruning pass.

IMPORTANT: qwen3_alps.py's get_qwen3() hardcodes model.seqlen=2048 (no CLI
override exists in that script) -- so despite the "nostrip8192" data file
name, the ORIGINAL ALPS baseline runs actually calibrated with 2048-token
windows, not 8192. This script must match that exactly (seqlen=2048) to
reproduce the same calibration the checkpoint being patched was built with.

Usage:
    python patch_alps_layer_sp0.py <pruned_model_path> <layer_idx> \
        --data_path <same data_path as original run> --save <out_path>
"""
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modelutils import find_layers
from alps import ALPS_prune
from qwen3_alps import get_ot_fw, _layer_fwd

DEV = torch.device('cuda')

PROJ_NAMES = [
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
]


class _StopFwd(Exception):
    pass


@torch.no_grad()
def capture_inps_for_layer(model, layer_idx, dataloader, nsamples, seqlen, dev):
    layer = model.model.layers[layer_idx]
    dtype = next(model.parameters()).dtype
    hidden = model.config.hidden_size
    inps = torch.zeros((nsamples, seqlen, hidden), dtype=dtype, device=dev)
    cache = {'attention_mask': None, 'position_ids': None, 'position_embeddings': None}
    state = {'i': 0}

    def hook(module, args, kwargs):
        hs = args[0] if args else kwargs['hidden_states']
        inps[state['i']].copy_(hs[0])
        for k in ('attention_mask', 'position_ids', 'position_embeddings'):
            if k in kwargs:
                cache[k] = kwargs[k]
        raise _StopFwd()

    h = layer.register_forward_pre_hook(hook, with_kwargs=True)
    for j, batch in enumerate(dataloader):
        state['i'] = j
        try:
            model(batch[0].to(dev))
        except _StopFwd:
            pass
    h.remove()
    return inps, cache


@torch.no_grad()
def run_layer_admm_sp0(layer, inps, cache, nsamples, rho):
    full = find_layers(layer)
    scd = {}
    for name in full:
        scd[name] = ALPS_prune(full[name], nsamples=nsamples, seqlen=inps.shape[1])

    def add_batch(name):
        def tmp(_, inp, out):
            scd[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = [full[name].register_forward_hook(add_batch(name)) for name in full]
    for j in range(nsamples):
        _layer_fwd(layer, inps[j].unsqueeze(0), cache)
    for h in handles:
        h.remove()

    for name in full:
        w = scd[name].layer.weight.data
        nnz_before = (w != 0).sum().item()
        total = w.numel()
        print(f'  {name}: sparsity before={1 - nnz_before/total:.4f}', flush=True)
        scd[name].ALPS_admm(sp=0.0, nm_n=0, nm_m=0, rho=rho)
        nnz_after = (scd[name].layer.weight.data != 0).sum().item()
        print(f'    -> sparsity after ALPS(sp=0.0) reconstruction: {1 - nnz_after/total:.4f}', flush=True)
        scd[name].free()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('pruned_model', type=str)
    p.add_argument('layer_idx', type=int)
    p.add_argument('--data_path', type=str, required=True)
    p.add_argument('--nsamples', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seqlen', type=int, default=2048,
                    help='must match the original run -- qwen3_alps.py hardcodes model.seqlen=2048, '
                         'ignore the "8192" in data filenames')
    p.add_argument('--rho', type=float, default=300.0)
    p.add_argument('--save', type=str, required=True)
    args = p.parse_args()

    print(f'Loading pruned model from {args.pruned_model} ...')
    model = AutoModelForCausalLM.from_pretrained(args.pruned_model, torch_dtype='auto').to(DEV)
    model.eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.pruned_model, trust_remote_code=True)

    print(f'Rebuilding the same calibration set (nsamples={args.nsamples}, seed={args.seed}, '
          f'seqlen={args.seqlen}) from {args.data_path} ...')
    dataloader, _ = get_ot_fw(args.nsamples, args.seed, args.seqlen, tokenizer, args.data_path)

    print(f'Capturing inputs to layer {args.layer_idx} (through the already-pruned layers before it)...')
    t0 = time.time()
    inps, cache = capture_inps_for_layer(model, args.layer_idx, dataloader, args.nsamples, args.seqlen, DEV)
    print(f'  captured in {time.time()-t0:.1f}s')

    print(f'Running ALPS Hessian collection + ADMM(sp=0.0) on layer {args.layer_idx}...')
    t0 = time.time()
    run_layer_admm_sp0(model.model.layers[args.layer_idx], inps, cache, args.nsamples, args.rho)
    print(f'  done in {time.time()-t0:.1f}s')

    print(f'Saving to {args.save} ...')
    model.save_pretrained(args.save)
    tokenizer.save_pretrained(args.save)
    print('Done.')
