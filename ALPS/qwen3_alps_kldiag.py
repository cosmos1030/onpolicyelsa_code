"""ALPS layer-wise KL diagnostic: for each individual Linear pruned (in ALPS's
own processing order -- one decoder layer at a time, all Linears within a
layer pruned sequentially against Hessian stats captured before any of them
were touched), run a full-model forward pass on a small held-out batch and
log two KL divergences:

  KL(dense || z_l)      -- cumulative drift from the original dense model
  KL(z_{l-1} || z_l)    -- incremental drift caused by this one Linear alone

alongside that Linear's own ALPS reconstruction error (already computed
inside ALPS_admm). Answers: does functional damage concentrate in specific
Linears at high sparsity (a "local reconstruction breaks down here" signal,
sharp incremental-KL spikes), or does it just accumulate smoothly across all
of them (a "TR/scope-widening story doesn't apply, it's just accumulation"
signal, flat incremental KL but rising cumulative KL)?

KL formula matches elsa/lib/gmp_trainer.py's _compute_tr_kl exactly:
KL(p||q) = sum_v p(v) * (log p(v) - log q(v)), mean over valid token
positions, log_softmax kept in bf16 with logits deleted immediately to avoid
holding multiple [B,T,V] float tensors at once.

Since a full model forward is needed after every single Linear regardless,
the whole model is kept resident on GPU throughout (no per-layer CPU
offloading like qwen3_alps.py's qwen3_sequential -- unnecessary memory
overhead for models that fit comfortably in bf16 on a single GPU, e.g. 4B).

Usage:
    python qwen3_alps_kldiag.py <model_path> <sparsity> --data_path <path> [--nsamples 32] [--heldout 8] [--seqlen 1024]
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from modelutils import find_layers
from alps import ALPS_prune
from qwen3_alps import get_qwen3, get_ot_fw, _make_catcher, _layer_fwd

DEV = torch.device('cuda')


@torch.no_grad()
def compute_logprobs(model, input_ids):
    """Full-model forward on a held-out batch -> log_softmax, kept in bf16."""
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids=input_ids).logits
    lp = F.log_softmax(logits[:, :-1, :], dim=-1)
    del logits
    return lp  # [B, T-1, V] bf16


@torch.no_grad()
def kl_div(lp_p, lp_q):
    """KL(p || q), mean over all token positions (held-out batch has no
    padding -- every position is valid, unlike the training-time TR check).
    Stays in bf16 throughout (matches gmp_trainer.py's _compute_tr_kl) --
    casting a [B,T,V] tensor to fp32 here (V=~150k) can transiently need
    tens of GB and OOM on a 24GB card; only the final reduced per-token KL
    is cast to float for the .item() call."""
    p = lp_p.exp()
    kl_tok = (p * (lp_p - lp_q)).sum(dim=-1)  # [B, T-1], bf16
    del p
    return max(kl_tok.float().mean().item(), 0.0)


@torch.no_grad()
def run_kldiag(model, dataloader, heldout_ids, args, log_path):
    print('Starting ALPS layer-wise KL diagnostic on Qwen3...')
    model.config.use_cache = False
    layers = model.model.layers
    model = model.to(DEV)

    dense_lp = compute_logprobs(model, heldout_ids)
    prev_lp = dense_lp
    print(f'Dense baseline computed on {heldout_ids.shape[0]} held-out samples x {heldout_ids.shape[1]} tokens.')

    nsamples = args.nsamples
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=DEV)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    Catcher = _make_catcher(inps, cache, nsamples)
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(DEV))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    records = []
    tot_params, tot_nnz = 0, 0

    for i in range(len(layers)):
        layer = layers[i]
        full = find_layers(layer)
        names = list(full.keys())
        subset = {n: full[n] for n in names}

        scd = {}
        for name in subset:
            scd[name] = ALPS_prune(subset[name], nsamples=nsamples, seqlen=model.seqlen)

        def add_batch(name):
            def tmp(_, inp, out):
                scd[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = [subset[name].register_forward_hook(add_batch(name)) for name in subset]
        for j in range(nsamples):
            _layer_fwd(layer, inps[j].unsqueeze(0), cache)
        for h in handles:
            h.remove()

        for name in subset:
            recon_error = scd[name].ALPS_admm(sp=args.sp, nm_n=args.nm_n, nm_m=args.nm_m, rho=args.rho)
            d1, d2 = scd[name].layer.weight.data.shape
            nnz = (scd[name].layer.weight.data.abs() > 0).sum().item()
            tot_params += d1 * d2
            tot_nnz += nnz
            scd[name].free()

            cur_lp = compute_logprobs(model, heldout_ids)
            kl_dense = kl_div(dense_lp, cur_lp)
            kl_incr = kl_div(prev_lp, cur_lp)
            del prev_lp
            prev_lp = cur_lp

            rec = {
                'layer': i, 'name': name, 'kl_dense': kl_dense,
                'kl_incremental': kl_incr, 'recon_error': recon_error,
            }
            records.append(rec)
            print(f'  layer={i:3d} {name:20s} kl_dense={kl_dense:.6f} kl_incr={kl_incr:.6f} recon_err={recon_error:.6f}')

            with open(log_path, 'a') as f:
                f.write(json.dumps(rec) + '\n')

        for j in range(nsamples):
            outs[j] = _layer_fwd(layer, inps[j].unsqueeze(0), cache)
        inps, outs = outs, inps
        torch.cuda.empty_cache()

    actual_sp = 1 - tot_nnz / tot_params
    print(f'ALPS KL diagnostic done. Actual sparsity: {actual_sp:.4f}')
    return records, actual_sp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('sp', type=float, help='Sparsity level')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=32, help='ALPS calibration samples (Hessian stats)')
    parser.add_argument('--heldout', type=int, default=8, help='held-out samples for the KL diagnostic forward passes')
    parser.add_argument('--seqlen', type=int, default=1024, help='overrides model.seqlen for this diagnostic (shorter = cheaper KL forwards)')
    parser.add_argument('--nm_n', type=int, default=0)
    parser.add_argument('--nm_m', type=int, default=0)
    parser.add_argument('--rho', type=float, default=300.0)
    parser.add_argument('--out', type=str, required=True, help='output .jsonl path for per-Linear records')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = get_qwen3(args.model)
    model.seqlen = args.seqlen
    model.eval()

    dataloader, _ = get_ot_fw(args.nsamples + args.heldout, args.seed, model.seqlen, tokenizer, args.data_path)
    calib_batches = dataloader[:args.nsamples]
    heldout_batches = dataloader[args.nsamples:args.nsamples + args.heldout]
    heldout_ids = torch.cat([b[0] for b in heldout_batches], dim=0).to(DEV)

    open(args.out, 'w').close()  # truncate/create

    tick = time.time()
    records, actual_sp = run_kldiag(model, calib_batches, heldout_ids, args, args.out)
    print(f'Total time: {time.time() - tick:.1f}s')
    print(f'Wrote {len(records)} records to {args.out}')
