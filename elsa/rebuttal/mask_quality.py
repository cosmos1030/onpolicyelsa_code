"""Rebuttal: quality of the target-reach mask vs the end-of-training mask.

Part A (dense + mask, no training): apply each mask to the ORIGINAL dense
Qwen3 weights and measure PPL (WikiText2 test, C4 val) and token-mean
KL(dense || masked) / PPL on held-out math CoT (math_220k_cot, not in SCOUT
training data). Masks: reach, final, and reach + random swap of the same
number of positions (controls for "any 0.016% change").

Part B (trained weights): take the FINAL trained model and zero out
  (i) the weights revived after reach,
  (ii) the same number of random surviving weights,
  (iii) the same number of smallest-|w| surviving weights.

Masks are read as the nonzero pattern of the two saved model dirs.
"""
import argparse, glob, gzip, json, math, random
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def load_masks(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k.endswith('.weight') and any(t in k for t in PROJ):
                    out[k] = (s.get_tensor(k) != 0)
    return out


def build_data(tok, seqlen, n_c4, n_math):
    from datasets import load_dataset
    sets = {}
    wt = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    ids = tok("\n\n".join(wt['text']), return_tensors='pt').input_ids[0]
    sets['wikitext2'] = [ids[i:i + seqlen] for i in range(0, ids.numel() - seqlen + 1, seqlen)]
    texts = []
    with gzip.open('/home1/doyoonkim/projects/elsa/data/c4/en/c4-validation.00000-of-00008.json.gz', 'rt') as f:
        for line in f:
            texts.append(json.loads(line)['text'])
            if len(texts) >= 5000:
                break
    ids = tok("\n\n".join(texts), return_tensors='pt').input_ids[0]
    sets['c4'] = [ids[i:i + seqlen] for i in range(0, ids.numel() - seqlen + 1, seqlen)][:n_c4]
    docs = []
    with open('/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl') as f:
        for line in f:
            t = json.loads(line)['text']
            ids = tok(t, return_tensors='pt').input_ids[0][:seqlen]
            if ids.numel() >= 512:
                docs.append(ids)
            if len(docs) >= n_math:
                break
    sets['mathcot'] = docs
    return sets


@torch.no_grad()
def evaluate(model, dense, sets, dev):
    res = {}
    for name, seqs in sets.items():
        nll = 0.0; kl = 0.0; ntok = 0
        for x in seqs:
            x = x.to(dev)[None]
            lg = model(input_ids=x).logits[0, :-1].float()
            lp = F.log_softmax(lg, -1); del lg
            tgt = x[0, 1:]
            nll += -lp.gather(-1, tgt[:, None]).sum().item()
            if dense is not None:
                dlp = F.log_softmax(dense(input_ids=x).logits[0, :-1].float(), -1)
                kl += (dlp.exp() * (dlp - lp)).sum().item()
                del dlp
            ntok += tgt.numel(); del lp
        res[name] = {'ppl': math.exp(nll / ntok), 'kl': kl / ntok if dense is not None else None}
    return res


def apply_mask(model, orig, masks):
    sd = dict(model.named_parameters())
    for k, m in masks.items():
        p = sd[k]
        p.data.copy_(orig[k].to(p.device) * m.to(p.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense', required=True)
    ap.add_argument('--reach', required=True)
    ap.add_argument('--final', required=True)
    ap.add_argument('--seqlen', type=int, default=2048)
    ap.add_argument('--n_c4', type=int, default=128)
    ap.add_argument('--n_math', type=int, default=128)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda'
    tok = AutoTokenizer.from_pretrained(a.dense)
    sets = build_data(tok, a.seqlen, a.n_c4, a.n_math)
    print({k: len(v) for k, v in sets.items()}, flush=True)

    dense = AutoModelForCausalLM.from_pretrained(a.dense, torch_dtype=torch.bfloat16).to(dev).eval()
    model = AutoModelForCausalLM.from_pretrained(a.dense, torch_dtype=torch.bfloat16).to(dev).eval()
    orig = {k: p.data.detach().cpu().clone() for k, p in model.named_parameters() if any(t in k for t in PROJ)}
    mr, mf = load_masks(a.reach), load_masks(a.final)
    keys = sorted(mr)
    tot = sum(mr[k].numel() for k in keys)
    n_rm = sum((mr[k] & ~mf[k]).sum().item() for k in keys)
    n_add = sum((~mr[k] & mf[k]).sum().item() for k in keys)
    print(f"prunable={tot:,} kept@reach={sum(mr[k].sum().item() for k in keys):,} "
          f"kept@final={sum(mf[k].sum().item() for k in keys):,} pruned_after_reach={n_rm:,} revived={n_add:,}", flush=True)

    results = {}

    def run(tag, masks, with_kl=True):
        apply_mask(model, orig, masks)
        r = evaluate(model, dense if with_kl else None, sets, dev)
        results[tag] = r
        print(tag, json.dumps(r), flush=True)
        json.dump(results, open(a.out, 'w'), indent=1)

    # ── Part A: dense weights + mask ──
    results['dense'] = evaluate(dense, None, sets, dev); print('dense', json.dumps(results['dense']), flush=True)
    run('A_reach', mr)
    run('A_final', mf)
    # random swap control: from the reach mask, prune n_rm random kept and revive n_add random pruned (global)
    sizes = [mr[k].numel() for k in keys]
    offs = [0];
    for s in sizes: offs.append(offs[-1] + s)
    flat_r = torch.cat([mr[k].flatten() for k in keys])
    kept_idx = flat_r.nonzero().squeeze(1); pr_idx = (~flat_r).nonzero().squeeze(1)
    for sd in range(a.seeds):
        g = torch.Generator().manual_seed(sd)
        f2 = flat_r.clone()
        f2[kept_idx[torch.randperm(kept_idx.numel(), generator=g)[:n_rm]]] = False
        f2[pr_idx[torch.randperm(pr_idx.numel(), generator=g)[:n_add]]] = True
        run(f'A_reach_randswap_s{sd}', {k: f2[offs[i]:offs[i + 1]].view_as(mr[k]) for i, k in enumerate(keys)})
        del f2
    del kept_idx, pr_idx

    # ── Part B: final trained weights, remove revived / controls ──
    fin = {}
    for f in sorted(glob.glob(f"{a.final}/*.safetensors")):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                fin[k] = s.get_tensor(k)
    model.load_state_dict({k: v for k, v in fin.items() if k in dict(model.named_parameters())}, strict=False)
    orig = {k: fin[k].clone() for k in keys}
    revived = {k: (~mr[k] & mf[k]) for k in keys}
    n_rev = sum(v.sum().item() for v in revived.values())
    run('B_final', mf)
    run('B_final_minus_revived', {k: mf[k] & ~revived[k] for k in keys})
    flat_f = torch.cat([mf[k].flatten() for k in keys]); flat_rev = torch.cat([revived[k].flatten() for k in keys])
    cand = (flat_f & ~flat_rev).nonzero().squeeze(1)
    for sd in range(a.seeds):
        g = torch.Generator().manual_seed(100 + sd)
        f2 = flat_f.clone(); f2[cand[torch.randperm(cand.numel(), generator=g)[:n_rev]]] = False
        run(f'B_final_minus_random_s{sd}', {k: f2[offs[i]:offs[i + 1]].view_as(mf[k]) for i, k in enumerate(keys)})
        del f2
    absw = torch.cat([orig[k].float().abs().flatten() for k in keys])
    absw[~flat_f | flat_rev] = float('inf')
    small = torch.topk(absw, n_rev, largest=False).indices
    f2 = flat_f.clone(); f2[small] = False
    run('B_final_minus_smallest_mag', {k: f2[offs[i]:offs[i + 1]].view_as(mf[k]) for i, k in enumerate(keys)})
    # per-type revived importance: mean |w| of revived vs all kept, in trained weights
    for k_t in PROJ:
        ks = [k for k in keys if f'.{k_t}.' in k]
        rv = torch.cat([orig[k].float().abs()[revived[k]] for k in ks]); kp = torch.cat([orig[k].float().abs()[mf[k]] for k in ks])
        print(f"  |w| {k_t:10s} revived mean={rv.mean().item() if rv.numel() else float('nan'):.5f} n={rv.numel():,}  kept mean={kp.mean().item():.5f} median={kp.median().item():.5f}", flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
