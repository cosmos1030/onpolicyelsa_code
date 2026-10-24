"""2:4 diagnostic: is dense->2:4 losing to ALPS because of WHICH weights it keeps
(saliency) or because nothing re-fits the surviving weights (compensation)?

Sequential one-shot 2:4 pruning of a Qwen3 model (same calibration and layer
loop as qwen3_alps.py). Each block's linears are masked with one criterion,
then either left as W*mask ("zero") or re-fit on that FIXED mask by masked
least squares against the dense layer's output (ALPS's own PCG back-solve,
"refit"). Logs per-linear relative reconstruction error and final PPL.

Criteria (the 2 kept per group of 4 consecutive input columns):
  magnitude  |w|
  wanda      |w| * ||x_j||
  fisher     F * w^2, F = squared NTP-loss gradient on the calibration set at
             the dense model -- what SCOUT's Adam exp_avg_sq starts from
  obs        w^2 / [H^-1]_jj (OBS/SparseGPT single-weight cost, no sequential update)
  alps       ALPS ADMM 2:4 (mask and weights from ALPS itself; refit is a no-op)
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from alps import ALPS_prune
from modelutils import find_layers
from qwen3_alps import get_ot_fw, get_qwen3, _make_catcher, _layer_fwd, qwen3_eval

DEV = torch.device('cuda')


def nm_keep(score, n=2, m=4):
    """Keep the top-n of every m consecutive input columns per row."""
    rows, cols = score.shape
    g = score[:, : cols // m * m].reshape(rows, cols // m, m)
    idx = g.topk(n, dim=-1).indices
    keep = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx, True)
    out = torch.ones_like(score, dtype=torch.bool)
    out[:, : cols // m * m] = keep.reshape(rows, -1)
    return out


def rel_err(W, Wd, H):
    D = (W - Wd).float()
    return float(((D @ H) * D).sum() / ((Wd.float() @ H) * Wd.float()).sum().clamp(min=1e-30))


def refit(alps, W_dense, keep):
    """Masked least squares on a fixed support, in ALPS's normalized system."""
    H = alps.XtX.clone().cpu()
    damp = 0.01 * torch.mean(torch.diag(H)).item()
    H[torch.arange(H.shape[0]), torch.arange(H.shape[0])] += damp
    xn = torch.diag(H).sqrt() + 1e-8
    Hn = (H / xn).T / xn
    Hn = Hn.T.contiguous()
    W = W_dense.float().cpu()
    YtX = (W * xn) @ Hn                                   # target: dense output
    B0 = ((W * keep.cpu()) * xn).t().to(DEV)
    supp = keep.t().float().to(DEV)
    B = alps.cg_batch(Hn.to(DEV), YtX.T.to(DEV), supp, M_bmm=None, X0=B0, rtol=1e-4, atol=0., maxiter=50)
    return (B.t() / xn.to(DEV)).to(W_dense.dtype)


def ntp_fisher(model, loader, nsamples):
    """Diagonal empirical Fisher (mean grad^2 of the NTP loss) for every linear in the blocks."""
    model.to(DEV)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    lin = {n: m for n, m in model.model.layers.named_modules() if isinstance(m, nn.Linear)}
    F = {n: torch.zeros_like(m.weight, dtype=torch.float32, device='cpu') for n, m in lin.items()}
    for p in model.parameters():
        p.requires_grad_(False)
    for m in lin.values():
        m.weight.requires_grad_(True)
    for j, (inp, _) in enumerate(loader[:nsamples]):
        model.zero_grad(set_to_none=True)
        out = model(inp.to(DEV), labels=inp.to(DEV))
        out.loss.backward()
        for n, m in lin.items():
            F[n] += m.weight.grad.float().pow(2).cpu()
    for m in lin.values():
        m.weight.requires_grad_(False)
        m.weight.grad = None
    model.gradient_checkpointing_disable()
    model.cpu()
    torch.cuda.empty_cache()
    return {n: f / nsamples for n, f in F.items()}


@torch.no_grad()
def run(model, loader, args, fisher):
    layers = model.model.layers
    nsamples = len(loader)
    model.model.embed_tokens.to(DEV)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb.to(DEV)
    layers[0].to(DEV)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=DEV)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}
    layers[0] = _make_catcher(inps, cache, nsamples)(layers[0])
    for b in loader:
        try:
            model(b[0].to(DEV))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0].cpu()
    model.model.embed_tokens.cpu()
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb.cpu()
    outs = torch.zeros_like(inps)
    recs = []
    for i in range(len(layers)):
        layer = layers[i].to(DEV)
        full = find_layers(layer)
        scd = {n: ALPS_prune(full[n], nsamples=nsamples, seqlen=model.seqlen) for n in full}
        hs = [full[n].register_forward_hook((lambda n: lambda _, a, o: scd[n].add_batch(a[0].data, o.data))(n)) for n in full]
        for j in range(nsamples):
            _layer_fwd(layer, inps[j].unsqueeze(0), cache)
        for h in hs:
            h.remove()
        for n in full:
            Wd = full[n].weight.data.clone()
            H = scd[n].XtX.clone().to(DEV)
            if args.criterion == 'alps':
                scd[n].ALPS_admm(sp=0.5, nm_n=2, nm_m=4, rho=300.0)
            else:
                w = Wd.float()
                if args.criterion == 'magnitude':
                    s = w.abs()
                elif args.criterion == 'wanda':
                    s = w.abs() * torch.diag(H).clamp(min=0).sqrt().reshape(1, -1)
                elif args.criterion == 'fisher':
                    s = fisher[f'{i}.{n}'].to(DEV) * w.pow(2)
                elif args.criterion == 'external':
                    s = None
                elif args.criterion == 'obs':
                    Hd = H.clone()
                    Hd[torch.arange(Hd.shape[0]), torch.arange(Hd.shape[0])] += 0.01 * torch.mean(torch.diag(Hd))
                    hinv = torch.diag(torch.cholesky_inverse(torch.linalg.cholesky(Hd.double()))).float()
                    s = w.pow(2) / hinv.reshape(1, -1)
                if args.criterion == 'external':
                    # zero pattern of another checkpoint's same layer, taken as the mask as-is
                    keep = (args._ext_get(f'model.layers.{i}.{n}.weight').to(DEV) != 0)
                else:
                    keep = nm_keep(s)
                full[n].weight.data = refit(scd[n], Wd, keep) if args.mode == 'refit' else Wd * keep
            Wp = full[n].weight.data
            Hf = H.float()
            recs.append({'layer': i, 'name': n, 'rel_err': rel_err(Wp, Wd, Hf),
                         'sparsity': float((Wp == 0).float().mean())})
            scd[n].free()
        e = [r['rel_err'] for r in recs if r['layer'] == i]
        print(f'layer {i}: mean rel_err {sum(e) / len(e):.5f}', flush=True)
        for j in range(nsamples):
            outs[j] = _layer_fwd(layer, inps[j].unsqueeze(0), cache)
        layers[i] = layer.cpu()
        del layer, scd
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--criterion', required=True, choices=['magnitude', 'wanda', 'fisher', 'obs', 'alps', 'external'])
    ap.add_argument('--mask_from', default='', help="--criterion external: checkpoint whose zero pattern is the mask")
    ap.add_argument('--save', default='', help='save the resulting pruned model here')
    ap.add_argument('--mode', default='zero', choices=['zero', 'refit'])
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--nsamples', type=int, default=128)
    ap.add_argument('--fisher_nsamples', type=int, default=64)
    ap.add_argument('--seqlen', type=int, default=2048)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = get_qwen3(args.model, seqlen=args.seqlen)
    model.eval()
    loader, testenc = get_ot_fw(args.nsamples, 42, args.seqlen, tok, args.data_path)
    if args.criterion == 'external':
        import glob as _g
        from safetensors import safe_open
        _idx = {}
        for f in _g.glob(os.path.join(args.mask_from, '*.safetensors')):
            with safe_open(f, 'pt') as g:
                for k in g.keys():
                    _idx[k] = f
        def _ext_get(k):
            with safe_open(_idx[k], 'pt') as g:
                return g.get_tensor(k)
        args._ext_get = _ext_get
    fisher = None
    if args.criterion == 'fisher':
        t0 = time.time()
        fisher = ntp_fisher(model, loader, args.fisher_nsamples)
        print(f'fisher from {args.fisher_nsamples} samples in {time.time() - t0:.0f}s', flush=True)
    t0 = time.time()
    recs = run(model, loader, args, fisher)
    ppl = qwen3_eval(model, testenc, DEV)
    res = {'criterion': args.criterion, 'mode': args.mode, 'ppl': ppl, 'time_s': round(time.time() - t0),
           'mean_rel_err': sum(r['rel_err'] for r in recs) / len(recs),
           'sparsity': sum(r['sparsity'] for r in recs) / len(recs), 'layers': recs}
    if args.save:
        model.save_pretrained(args.save)
        tok.save_pretrained(args.save)
        print(f'Saved pruned model to {args.save}', flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, 'w'))
    print(f"RESULT {args.criterion} {args.mode}: PPL {ppl:.3f} mean_rel_err {res['mean_rel_err']:.5f} sparsity {res['sparsity']:.4f}", flush=True)


if __name__ == '__main__':
    main()
