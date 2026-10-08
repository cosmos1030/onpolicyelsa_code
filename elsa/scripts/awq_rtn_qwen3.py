"""AWQ (activation-aware scaling + output-error clipping) and plain RTN for Qwen3,
as fake-quant checkpoints (no packing), to compare quantization starting points:

  --out_rtn        RTN of the original weights (per-group asymmetric min/max grid)
  --out_awq_dense  AWQ-transformed full-precision model (scales folded into the
                   preceding RMSNorm / up_proj, weights clipped): the same function
                   up to bf16 rounding and the clipping. Start point for SCOUT.
  --out_awq_rtn    RTN of the AWQ-transformed weights

Grid = lib.quant_commit.grid_params(mse_grid=0) + fake_quant, i.e. exactly the
grid SCOUT and the GPTQ baselines use. Per decoder layer, AWQ searches one
scale vector per input group (alpha in [0, 1], s = mean|x|^alpha normalised)
for: qkv (input_layernorm -> q/k/v), gate/up (post_attention_layernorm ->
gate/up), down (up_proj -> down_proj). v -> o is skipped as in AutoAWQ: under
GQA o_proj's input width differs from v_proj's output. Clipping is searched per
(output row, group) on every linear except q/k, minimising output error.
Scales and clips are fitted on the full-precision layer, and the next layer's
input is the full-precision output, as in AutoAWQ.
"""
import argparse, json, os, random, sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/home1/doyoonkim/projects/elsa')
from lib.quant_commit import grid_params, fake_quant


def calib_windows(path, tok, n, seqlen, seed):
    rows = [json.loads(l)['text'] for l in open(path)]
    rng = random.Random(seed)
    out = []
    for i in rng.sample(range(len(rows)), len(rows)):
        ids = tok(rows[i]).input_ids
        if len(ids) >= seqlen:
            out.append(torch.tensor(ids[:seqlen]))
        if len(out) == n:
            break
    return torch.stack(out)


def rtn(w, bits, gs):
    s, z = grid_params(w.float(), bits, gs, False, mse_grid=0)
    return fake_quant(w.float(), s, z, bits, False).to(w.dtype)


def sub_tokens(x, k, gen):
    x = x.reshape(-1, x.shape[-1])
    idx = torch.randperm(x.shape[0], generator=gen, device='cpu')[:k].to(x.device)
    return x[idx].float()


@torch.no_grad()
def search_scale(x, x_absmean, weights, bits, gs, n_grid=20):
    """Best per-input-channel scale for a group of linears sharing input x."""
    ref = [x @ w.float().t() for w in weights]
    best, best_s = float('inf'), None
    for i in range(n_grid + 1):
        a = i / n_grid
        s = x_absmean.clamp(min=1e-4).pow(a)
        s = s / (s.max() * s.min()).sqrt()
        err = 0.0
        for w, r in zip(weights, ref):
            wq = rtn(w.float() * s, bits, gs).float() / s
            err += float(((x @ wq.t()) - r).pow(2).mean())
        if err < best:
            best, best_s = err, s.clone()
    return best_s


@torch.no_grad()
def search_clip(x, w, bits, gs, n_grid=20, max_shrink=0.5, row_chunk=512):
    """Per (row, group) clip ratio minimising that group's output contribution error."""
    o, i = w.shape
    ng = i // gs
    xg = x.view(-1, ng, gs)                                    # T, ng, gs
    wg = w.float().view(o, ng, gs)
    best_ratio = torch.ones(o, ng, 1, device=w.device)
    for r0 in range(0, o, row_chunk):
        wc = wg[r0:r0 + row_chunk]
        ref = torch.einsum('tgk,ogk->tog', xg, wc)
        best_err = torch.full(wc.shape[:2], float('inf'), device=w.device)
        mx, mn = wc.amax(-1, keepdim=True), wc.amin(-1, keepdim=True)
        for j in range(n_grid):
            ratio = 1 - max_shrink * j / n_grid
            wcl = torch.max(torch.min(wc, mx * ratio), mn * ratio)
            wq = rtn(wcl.reshape(-1, i), bits, gs).float().view_as(wc)
            err = (torch.einsum('tgk,ogk->tog', xg, wq) - ref).pow(2).mean(0)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_ratio[r0:r0 + row_chunk] = torch.where(better.unsqueeze(-1), torch.tensor(ratio, device=w.device),
                                                        best_ratio[r0:r0 + row_chunk])
    mx, mn = wg.amax(-1, keepdim=True), wg.amin(-1, keepdim=True)
    return torch.max(torch.min(wg, mx * best_ratio), mn * best_ratio).view(o, i).to(w.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--bits', type=int, default=3)
    ap.add_argument('--group', type=int, default=128)
    ap.add_argument('--data_path', default='/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl')
    ap.add_argument('--nsamples', type=int, default=64)
    ap.add_argument('--seqlen', type=int, default=2048)
    ap.add_argument('--n_tok', type=int, default=4096, help='tokens sampled per input for the search losses')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_clip', action='store_true')
    ap.add_argument('--out_rtn', default='')
    ap.add_argument('--out_awq_dense', default='')
    ap.add_argument('--out_awq_rtn', default='')
    args = ap.parse_args()
    dev = torch.device('cuda')
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev).eval()
    model.config.use_cache = False
    layers = model.model.layers
    bits, gs = args.bits, args.group

    def quantize_all(m):
        for l in m.model.layers:
            for mod in l.modules():
                if isinstance(mod, nn.Linear):
                    mod.weight.data = rtn(mod.weight.data, bits, gs)

    if args.out_rtn:                                  # plain RTN of the original weights
        rtn_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        quantize_all(rtn_model)
        rtn_model.save_pretrained(args.out_rtn); tok.save_pretrained(args.out_rtn)
        del rtn_model
        print('saved RTN ->', args.out_rtn, flush=True)

    if not (args.out_awq_dense or args.out_awq_rtn):
        return
    X = calib_windows(args.data_path, tok, args.nsamples, args.seqlen, args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    cache = {}
    class Catcher(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m
        def __getattr__(self, name):              # the model reads e.g. layer.attention_type
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.m, name)
        def forward(self, x, **kw):
            cache.setdefault('x', []).append(x); cache['kw'] = kw
            raise ValueError
    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for b in X:
            try:
                model(b.unsqueeze(0).to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].m
    hs, kw = cache['x'], cache['kw']

    for li, layer in enumerate(layers):
        feats = {}
        def grab(name):
            def h(mod, inp, out):
                x = inp[0].detach()
                d = feats.setdefault(name, {'sum': 0, 'n': 0, 'tok': []})
                d['sum'] = d['sum'] + x.abs().float().reshape(-1, x.shape[-1]).sum(0)
                d['n'] += x.shape[0] * x.shape[1]
                d['tok'].append(sub_tokens(x, max(1, args.n_tok // len(hs)), gen))
            return h
        hooks = [layer.self_attn.q_proj.register_forward_hook(grab('qkv')),
                 layer.mlp.gate_proj.register_forward_hook(grab('gateup')),
                 layer.mlp.down_proj.register_forward_hook(grab('down'))]
        with torch.no_grad():
            for x in hs:
                layer(x, **kw)
        for h in hooks:
            h.remove()
        F = {k: (v['sum'] / v['n'], torch.cat(v['tok'])) for k, v in feats.items()}
        at, mlp = layer.self_attn, layer.mlp
        with torch.no_grad():
            s = search_scale(F['qkv'][1], F['qkv'][0], [at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], bits, gs)
            layer.input_layernorm.weight.div_(s.to(layer.input_layernorm.weight.dtype))
            for lin in (at.q_proj, at.k_proj, at.v_proj):
                lin.weight.mul_(s.to(lin.weight.dtype))
            s = search_scale(F['gateup'][1], F['gateup'][0], [mlp.gate_proj.weight, mlp.up_proj.weight], bits, gs)
            layer.post_attention_layernorm.weight.div_(s.to(layer.post_attention_layernorm.weight.dtype))
            for lin in (mlp.gate_proj, mlp.up_proj):
                lin.weight.mul_(s.to(lin.weight.dtype))
            s = search_scale(F['down'][1], F['down'][0], [mlp.down_proj.weight], bits, gs)
            mlp.up_proj.weight.div_(s.to(mlp.up_proj.weight.dtype).unsqueeze(1))
            mlp.down_proj.weight.mul_(s.to(mlp.down_proj.weight.dtype))
        if not args.no_clip:                          # inputs as seen by the scaled layer
            feats = {}
            names = {'v': at.v_proj, 'o': at.o_proj, 'gate': mlp.gate_proj, 'up': mlp.up_proj, 'down': mlp.down_proj}
            hooks = [m.register_forward_hook(grab(n)) for n, m in names.items()]
            with torch.no_grad():
                for x in hs:
                    layer(x, **kw)
            for h in hooks:
                h.remove()
            with torch.no_grad():
                for n, m in names.items():
                    m.weight.data = search_clip(torch.cat(feats[n]['tok']), m.weight.data, bits, gs)
        with torch.no_grad():
            nxt = []
            for x in hs:
                o = layer(x, **kw)
                nxt.append(o[0] if isinstance(o, tuple) else o)
            hs = nxt
        del feats, F
        torch.cuda.empty_cache()
        print(f'layer {li} done', flush=True)

    model = model.cpu()
    if args.out_awq_dense:
        model.save_pretrained(args.out_awq_dense); tok.save_pretrained(args.out_awq_dense)
        print('saved AWQ dense ->', args.out_awq_dense, flush=True)
    if args.out_awq_rtn:
        quantize_all(model)
        model.save_pretrained(args.out_awq_rtn); tok.save_pretrained(args.out_awq_rtn)
        print('saved AWQ+RTN ->', args.out_awq_rtn, flush=True)


if __name__ == '__main__':
    main()
