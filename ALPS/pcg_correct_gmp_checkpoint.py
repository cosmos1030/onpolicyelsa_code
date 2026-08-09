"""Apply an ALPS-style PCG backsolve correction to an already-pruned GMP/TR-GMP
checkpoint, WITHOUT re-deriving the mask via ADMM -- the sparsity pattern is
taken as-is from the checkpoint. Only the surviving (nonzero) weights are
re-solved via conjugate gradient to better reconstruct the DENSE reference
model's per-layer input/output function, using the same sequential
layer-wise calibration pipeline as qwen3_alps.py (each layer's calibration
input X comes from the actual pruned network's forward pass, propagated
through already-corrected earlier layers -- not from the dense model).

For each Linear submodule:
    X          = calibration input activations (from the PRUNED model's own
                 forward pass, so it reflects upstream pruning/correction)
    W_dense    = the ORIGINAL dense checkpoint's weight for this submodule
                 (reconstruction target: Y = X @ W_dense^T)
    W_pruned   = the current (sparse) weight -- defines the fixed support
                 and the PCG warm start
    -> solve   min_W ||XW^T - XW_dense^T||^2  s.t. support(W) = support(W_pruned)
       via ALPS_prune.cg_batch (same routine ALPS uses for its own backsolve
       step, just reused here directly with maxiter=10, no ADMM search).

Usage:
    python pcg_correct_gmp_checkpoint.py <dense_model_path> <pruned_model_path_or_hf_id> \
        --data_path <calib.jsonl> --save <out_dir> [--nsamples 128] [--push_to_hub <repo_id>]
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from modelutils import find_layers
from alps import ALPS_prune
from qwen3_alps import get_ot_fw, _make_catcher, _layer_fwd

DEV = torch.device('cuda:0')


def get_model(path):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype='auto', trust_remote_code=True)
    model.seqlen = 2048
    return model


@torch.no_grad()
def pcg_correct(pruned_model, dense_model, dataloader, dev, nsamples):
    print('Starting PCG-only correction (mask fixed from checkpoint)...')

    use_cache = pruned_model.config.use_cache
    pruned_model.config.use_cache = False
    layers = pruned_model.model.layers
    dense_layers = dense_model.model.layers

    dtype = next(iter(pruned_model.parameters())).dtype
    hidden = pruned_model.config.hidden_size
    inps = torch.zeros((nsamples, pruned_model.seqlen, hidden), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    pruned_model.model.embed_tokens = pruned_model.model.embed_tokens.to(dev)
    if hasattr(pruned_model.model, 'norm'):
        pruned_model.model.norm = pruned_model.model.norm.to(dev)
    if hasattr(pruned_model.model, 'rotary_emb'):
        pruned_model.model.rotary_emb = pruned_model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    Catcher = _make_catcher(inps, cache, nsamples)
    layers[0] = Catcher(layers[0])
    for inp, _ in dataloader:
        try:
            pruned_model(inp.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    pruned_model.model.embed_tokens = pruned_model.model.embed_tokens.cpu()
    if hasattr(pruned_model.model, 'rotary_emb'):
        pruned_model.model.rotary_emb = pruned_model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    cg_helper = object.__new__(ALPS_prune)  # only used to call cg_batch, no __init__ needed

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        dense_layer = dense_layers[i].to(dev)
        pruned_sub = find_layers(layer)
        dense_sub = find_layers(dense_layer)

        xtx = {name: torch.zeros((m.weight.shape[1], m.weight.shape[1]), device=dev, dtype=torch.float32)
               for name, m in pruned_sub.items()}
        captured = {name: [] for name in pruned_sub}

        def add_hook(name):
            def hook(_, inp, out):
                x = inp[0]
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                x = x.float().t()  # [in_features, tokens]
                xtx[name] += x @ x.t()
            return hook

        handles = [pruned_sub[name].register_forward_hook(add_hook(name)) for name in pruned_sub]
        for j in range(nsamples):
            _layer_fwd(layer, inps[j].unsqueeze(0), cache)
        for h in handles:
            h.remove()

        for name in pruned_sub:
            t0 = time.time()
            W_pruned = pruned_sub[name].weight.data.clone().float().to(dev)
            W_dense = dense_sub[name].weight.data.clone().float().to(dev)

            XtX = xtx[name]
            damp = 0.01 * torch.mean(torch.diag(XtX)).item()
            diag = torch.arange(XtX.shape[0], device=dev)
            XtX[diag, diag] += damp
            X_norm = torch.diag(XtX).sqrt() + 1e-8
            XtX = XtX / X_norm
            XtX = (XtX.T / X_norm).T

            YtX = torch.matmul(W_dense * X_norm, XtX)  # dense reconstruction target
            B0 = (W_pruned * X_norm).t().contiguous()  # warm start = current sparse weight
            A_supp = (B0 != 0).float()

            B = ALPS_prune.cg_batch(
                cg_helper, XtX, YtX.t(), A_supp, M_bmm=None, X0=B0,
                rtol=1e-4, atol=0., maxiter=10, verbose=False,
            )
            new_w = (B.t() / X_norm).reshape(W_pruned.shape).to(dtype)
            pruned_sub[name].weight.data.copy_(new_w)

            nnz_before = (W_pruned != 0).sum().item()
            nnz_after = (new_w != 0).sum().item()
            print(f'  layer {i} {name}: PCG done in {time.time()-t0:.1f}s, '
                  f'nnz {nnz_before}->{nnz_after} (support unchanged: {nnz_before == nnz_after})',
                  flush=True)

        for j in range(nsamples):
            outs[j] = _layer_fwd(layer, inps[j].unsqueeze(0), cache)

        layers[i] = layer.cpu()
        dense_layers[i] = dense_layer.cpu()
        del layer, dense_layer, xtx, captured
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    pruned_model.config.use_cache = use_cache
    print('PCG correction done.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dense_model_path')
    p.add_argument('pruned_model_path')
    p.add_argument('--data_path', required=True)
    p.add_argument('--nsamples', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save', required=True)
    p.add_argument('--push_to_hub', default=None)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'Loading dense model from {args.dense_model_path} ...')
    dense_model = get_model(args.dense_model_path)
    dense_model.eval()

    print(f'Loading pruned model from {args.pruned_model_path} ...')
    pruned_model = get_model(args.pruned_model_path)
    pruned_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.pruned_model_path, use_fast=False, trust_remote_code=True)

    print(f'Loading calibration data from {args.data_path} ({args.nsamples} samples)...')
    dataloader, _ = get_ot_fw(args.nsamples, args.seed, pruned_model.seqlen, tokenizer, args.data_path)

    pcg_correct(pruned_model, dense_model, dataloader, DEV, args.nsamples)

    print(f'Saving corrected model to {args.save} ...')
    pruned_model.save_pretrained(args.save)
    tokenizer.save_pretrained(args.save)

    if args.push_to_hub:
        print(f'Pushing to {args.push_to_hub} ...')
        pruned_model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)
        print(f'Pushed to https://huggingface.co/{args.push_to_hub}')


if __name__ == '__main__':
    main()
