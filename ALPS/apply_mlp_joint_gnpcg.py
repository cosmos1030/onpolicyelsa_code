"""Apply MLP-block joint Gauss-Newton PCG refinement (see mlp_joint_gnpcg.py)
to EVERY decoder layer of an already-pruned ALPS checkpoint, sequentially
(each layer's calibration input is the ALREADY-refined network's own
propagated hidden state, same convention as ALPS's own PCG backsolve in
pcg_correct_gmp_checkpoint.py), and save the resulting full model checkpoint
for downstream eval (PPL/zero-shot/reasoning via eval_full.py).

Logs per-layer calib/held-out recon RMS (before and after refinement) so the
trajectory is inspectable without needing a separate run.

Usage:
    python apply_mlp_joint_gnpcg.py <dense_model_path> <alps_model_path> \\
        --data_path <calib.jsonl> --save <out_dir> \\
        --pcg_iters 10 --gn_outer 1 --damping 1e-3 \\
        --nsamples 128 --heldout 64 --seqlen 2048 [--push_to_hub <repo_id>]
"""
import argparse
import random
import time

import numpy as np
import torch

from qwen3_alps import get_ot_fw, _make_catcher, _layer_fwd
from mlp_joint_gnpcg import (
    get_model, block_forward, gn_pcg_solve, recon_error, _tree_add,
)

DEV = torch.device('cuda:0')


@torch.no_grad()
def refine_all_layers(alps_model, dense_model, calib_loader, heldout_loader,
                       dev, nsamples, heldout_n, seqlen, pcg_iters, gn_outer, damping):
    use_cache = alps_model.config.use_cache
    alps_model.config.use_cache = False
    layers = alps_model.model.layers
    dense_layers = dense_model.model.layers

    dtype = next(iter(alps_model.parameters())).dtype
    hidden = alps_model.config.hidden_size

    calib_inps = torch.zeros((nsamples, seqlen, hidden), dtype=dtype, device=dev)
    held_inps = torch.zeros((heldout_n, seqlen, hidden), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    alps_model.model.embed_tokens = alps_model.model.embed_tokens.to(dev)
    if hasattr(alps_model.model, 'rotary_emb'):
        alps_model.model.rotary_emb = alps_model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    Catcher = _make_catcher(calib_inps, cache, nsamples)
    layers[0] = Catcher(layers[0])
    for inp, _ in calib_loader:
        try:
            alps_model(inp.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    cache_h = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}
    Catcher_h = _make_catcher(held_inps, cache_h, heldout_n)
    layers[0] = Catcher_h(layers[0])
    for inp, _ in heldout_loader:
        try:
            alps_model(inp.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    alps_model.model.embed_tokens = alps_model.model.embed_tokens.cpu()
    if hasattr(alps_model.model, 'rotary_emb'):
        alps_model.model.rotary_emb = alps_model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    calib_outs = torch.zeros_like(calib_inps)
    held_outs = torch.zeros_like(held_inps)

    trajectory = []

    for i in range(len(layers)):
        t0 = time.time()
        layer = layers[i].to(dev)
        dense_layer = dense_layers[i].to(dev)

        # Capture this layer's MLP input/output on both calib and held-out sets.
        captured = {'calib_in': [], 'calib_out': [], 'held_in': [], 'held_out': []}

        def hook_calib(_m, inp, out):
            captured['calib_in'].append(inp[0].detach())
            captured['calib_out'].append(out.detach())

        def hook_held(_m, inp, out):
            captured['held_in'].append(inp[0].detach())
            captured['held_out'].append(out.detach())

        h = layer.mlp.register_forward_hook(hook_calib)
        for j in range(nsamples):
            calib_outs[j] = _layer_fwd(layer, calib_inps[j].unsqueeze(0), cache)
        h.remove()

        h = layer.mlp.register_forward_hook(hook_held)
        for j in range(heldout_n):
            held_outs[j] = _layer_fwd(layer, held_inps[j].unsqueeze(0), cache_h)
        h.remove()

        X_calib = torch.cat(captured['calib_in'], dim=0).reshape(-1, hidden).float()
        X_held = torch.cat(captured['held_in'], dim=0).reshape(-1, hidden).float()

        dense_params = {
            'gate': dense_layer.mlp.gate_proj.weight.detach().clone().float(),
            'up': dense_layer.mlp.up_proj.weight.detach().clone().float(),
            'down': dense_layer.mlp.down_proj.weight.detach().clone().float(),
        }
        with torch.no_grad():
            Y_ref_calib = block_forward(dense_params, X_calib)
            Y_ref_held = block_forward(dense_params, X_held)

        params = {
            'gate': layer.mlp.gate_proj.weight.detach().clone().float(),
            'up': layer.mlp.up_proj.weight.detach().clone().float(),
            'down': layer.mlp.down_proj.weight.detach().clone().float(),
        }
        masks = {k: (v != 0).float() for k, v in params.items()}

        e0_calib = recon_error(params, X_calib, Y_ref_calib)
        e0_held = recon_error(params, X_held, Y_ref_held)

        cur = {k: v.clone() for k, v in params.items()}
        done = 0
        for outer in range(gn_outer):
            remaining = max(pcg_iters // gn_outer, 1) if outer < gn_outer - 1 else pcg_iters - done
            remaining = max(remaining, 1)
            delta = gn_pcg_solve(cur, X_calib, Y_ref_calib, masks, damping, remaining)
            cur = _tree_add(cur, delta, alpha=1.0)
            done += remaining

        e1_calib = recon_error(cur, X_calib, Y_ref_calib)
        e1_held = recon_error(cur, X_held, Y_ref_held)

        layer.mlp.gate_proj.weight.data.copy_(cur['gate'].to(dtype))
        layer.mlp.up_proj.weight.data.copy_(cur['up'].to(dtype))
        layer.mlp.down_proj.weight.data.copy_(cur['down'].to(dtype))

        # Re-run forward with refined weights to get correct propagated outputs
        # for the NEXT layer's input (calib and held-out).
        for j in range(nsamples):
            calib_outs[j] = _layer_fwd(layer, calib_inps[j].unsqueeze(0), cache)
        for j in range(heldout_n):
            held_outs[j] = _layer_fwd(layer, held_inps[j].unsqueeze(0), cache_h)

        dt = time.time() - t0
        print(f'layer {i:2d} ({dt:5.1f}s): calib_rms {e0_calib:.5f} -> {e1_calib:.5f}  '
              f'held_rms {e0_held:.5f} -> {e1_held:.5f}', flush=True)
        trajectory.append({'layer': i, 'calib_rms_before': e0_calib, 'calib_rms_after': e1_calib,
                            'held_rms_before': e0_held, 'held_rms_after': e1_held, 'sec': dt})

        layers[i] = layer.cpu()
        dense_layers[i] = dense_layer.cpu()
        del layer, dense_layer, captured, X_calib, X_held, dense_params, params, cur
        torch.cuda.empty_cache()
        calib_inps, calib_outs = calib_outs, calib_inps
        held_inps, held_outs = held_outs, held_inps

    alps_model.config.use_cache = use_cache
    return trajectory


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dense_model_path')
    p.add_argument('alps_model_path')
    p.add_argument('--data_path', required=True)
    p.add_argument('--save', required=True)
    p.add_argument('--pcg_iters', type=int, default=10)
    p.add_argument('--gn_outer', type=int, default=1)
    p.add_argument('--damping', type=float, default=1e-3)
    p.add_argument('--nsamples', type=int, default=128)
    p.add_argument('--heldout', type=int, default=64)
    p.add_argument('--seqlen', type=int, default=2048)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--push_to_hub', default=None)
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

    trajectory = refine_all_layers(
        alps_model, dense_model, calib_loader, heldout_loader, DEV,
        args.nsamples, args.heldout, args.seqlen,
        args.pcg_iters, args.gn_outer, args.damping,
    )

    print('\n=== summary ===')
    tot_before = sum(t['held_rms_before'] for t in trajectory)
    tot_after = sum(t['held_rms_after'] for t in trajectory)
    print(f'sum(held_rms) before={tot_before:.4f} after={tot_after:.4f} '
          f'({"IMPROVED" if tot_after < tot_before else "WORSE"})')

    print(f'Saving refined model to {args.save} ...')
    alps_model.save_pretrained(args.save)
    tokenizer.save_pretrained(args.save)

    if args.push_to_hub:
        print(f'Pushing to {args.push_to_hub} ...')
        alps_model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f'Pushed to https://huggingface.co/{args.push_to_hub}')


if __name__ == '__main__':
    main()
