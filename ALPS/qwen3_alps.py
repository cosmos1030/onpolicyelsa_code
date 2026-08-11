import argparse
import random
import subprocess
import sys
import time
import os
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelutils import find_layers
from alps import ALPS_prune

EVAL_FULL_SCRIPT = "/home1/doyoonkim/projects/elsa/scripts/eval_full.py"

DEV = torch.device('cuda')


# ── Data ─────────────────────────────────────────────────────────────────────

def get_ot_fw(nsamples, seed, seqlen, tokenizer, data_path):
    raw = load_dataset('json', data_files=data_path, split='train')
    random.seed(seed)
    np.random.seed(seed)

    all_tokens = []
    for sample in raw:
        text = sample.get('text', '')
        if not text:
            continue
        tokens = tokenizer(text, return_tensors='pt').input_ids
        if tokens.shape[1] >= seqlen:
            all_tokens.append(tokens)
    assert len(all_tokens) > 0, "No samples longer than seqlen"

    trainloader = []
    for _ in range(nsamples):
        src = random.choice(all_tokens)
        i = random.randint(0, src.shape[1] - seqlen)
        inp = src[:, i:i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    full_text = ' '.join(s.get('text', '') for s in list(raw)[:500])
    testenc = tokenizer(full_text, return_tensors='pt')

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    return trainloader, TokenizerWrapper(testenc.input_ids[:, :256 * seqlen])


# ── Model ─────────────────────────────────────────────────────────────────────

def get_qwen3(model_path):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype='auto', trust_remote_code=True
    )
    model.seqlen = 2048
    return model


def _make_catcher(inps, cache, nsamples):
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            if cache['i'] < nsamples:
                inps[cache['i']] = inp if not isinstance(inp, tuple) else inp[0]
            cache['i'] += 1
            for key in ('attention_mask', 'position_ids', 'position_embeddings'):
                if key in kwargs:
                    cache[key] = kwargs[key]
            raise ValueError

    return Catcher


def _layer_fwd(layer, inp, cache):
    kwargs = {}
    for key in ('attention_mask', 'position_ids', 'position_embeddings'):
        if cache.get(key) is not None:
            kwargs[key] = cache[key]
    return layer(inp, **kwargs)[0]


# ── Pruning ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def qwen3_sequential(model, dataloader, dev, args):
    print('Starting ALPS on Qwen3...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, 'norm'):
        model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    nsamples = args.nsamples
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    Catcher = _make_catcher(inps, cache, nsamples)
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    print('Ready.')

    tot_params, tot_nnz = 0, 0

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]

        scd = {}
        for names in sequential:
            subset = {n: full[n] for n in names}
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
                print(f'  Layer {i} {name}')
                scd[name].ALPS_admm(sp=args.sp, nm_n=args.nm_n, nm_m=args.nm_m, rho=args.rho)
                d1, d2 = scd[name].layer.weight.data.shape
                nnz = (scd[name].layer.weight.data.abs() > 0).sum().item()
                tot_params += d1 * d2
                tot_nnz += nnz
                scd[name].free()

        for j in range(nsamples):
            outs[j] = _layer_fwd(layer, inps[j].unsqueeze(0), cache)

        layers[i] = layer.cpu()
        del layer, scd
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    actual_sp = 1 - tot_nnz / tot_params
    print(f'ALPS pruning done. Actual sparsity: {actual_sp:.4f}')
    return actual_sp


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def qwen3_eval(model, testenc, dev):
    print('Evaluating...')
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    Catcher = _make_catcher(inps, cache, nsamples)
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, i * model.seqlen:(i + 1) * model.seqlen].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = _layer_fwd(layer, inps[j].unsqueeze(0), cache)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if hasattr(model.model, 'norm') and model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0).to(dev)
        if hasattr(model.model, 'norm') and model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, i * model.seqlen:(i + 1) * model.seqlen][:, 1:].to(dev)
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * model.seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f'PPL: {ppl.item():.4f}')
    model.config.use_cache = use_cache
    return ppl.item()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('sp', type=float, help='Sparsity level')
    parser.add_argument('--data_path', type=str, required=True, help='Path to ot+fw JSONL')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--nm_n', type=int, default=0)
    parser.add_argument('--nm_m', type=int, default=0)
    parser.add_argument('--rho', type=float, default=300.0)
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--eval_full', action='store_true', help='Run full eval (PPL+zeroshot+lighteval) after pruning')
    parser.add_argument('--profile', type=str, default='full', choices=['full', 'quick'],
                         help="lighteval profile passed through to eval_full.py: 'full' (32768/38912 budget, "
                              "incl. AIME24/25) or 'quick' (8192 budget, no AIME -- for ranking configs cheaply)")
    parser.add_argument('--wandb_project', type=str, default='reasoning_qwen3_1.7b')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--gpu_util', type=float, default=0.9)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--out_base', type=str, default='')
    parser.add_argument('--push_to_hub', action='store_true', help='Upload pruned model to HuggingFace Hub after saving')
    parser.add_argument('--hub_model_id', type=str, default=None, help='HF Hub repo id (e.g. username/model-name); auto-generated if not given')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = get_qwen3(args.model)
    model.eval()

    # One-shot calibration is forward-only: ~2*N*tokens (no backward pass),
    # unlike gradient fine-tuning's ~6*N*tokens.
    n_params = sum(p.numel() for p in model.parameters())
    n_tokens = args.nsamples * model.seqlen
    flops = 2 * n_params * n_tokens
    print(f'Calibration FLOPs: {flops:.3e} ({n_params} params x {n_tokens} tokens, forward-only)')

    dataloader, testenc = get_ot_fw(args.nsamples, args.seed, model.seqlen, tokenizer, args.data_path)

    tick = time.time()
    qwen3_sequential(model, dataloader, DEV, args)
    print(f'Pruning time: {time.time() - tick:.1f}s')

    ppl = qwen3_eval(model, testenc, DEV)
    print(f'Final PPL: {ppl:.4f}')

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)
        print(f'Saved to {args.save}')

    hub_model_id = None
    hub_url = None
    if args.push_to_hub and args.save:
        try:
            from huggingface_hub import HfApi
            for _env in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                os.environ.pop(_env, None)
            try:
                import huggingface_hub.constants as _hf_const
                _hf_const.HF_HUB_OFFLINE = False
            except Exception:
                pass
            hub_model_id = args.hub_model_id
            if not hub_model_id:
                from datetime import datetime as _dt
                _now = _dt.now().strftime("%Y%m%d_%H%M%S")
                hub_model_id = f"cosmos1030/alps-s{int(args.sp * 100)}pct_{_now}"
            print(f'Uploading model to HuggingFace Hub: {hub_model_id}')
            api = HfApi()
            api.create_repo(repo_id=hub_model_id, exist_ok=True)
            api.upload_folder(
                folder_path=args.save,
                repo_id=hub_model_id,
                commit_message=f"ALPS pruned: sparsity={args.sp}",
            )
            hub_url = f"https://huggingface.co/{hub_model_id}"
            print(f'Uploaded to {hub_url}')
        except Exception as e:
            print(f'WARNING: push_to_hub failed ({e}); continuing without upload.')
            hub_model_id = None
            hub_url = None

    if args.eval_full:
        if not args.save:
            print('WARNING: --eval_full requires --save; skipping full eval')
        else:
            # Release GPU memory before launching vLLM in eval_full subprocess
            import gc
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print('GPU memory released before eval_full subprocess')

            run_name = args.run_name or f'alps_s{int(args.sp * 100)}pct'
            cmd = [
                sys.executable, EVAL_FULL_SCRIPT,
                '--model_path', args.save,
                '--wandb_project', args.wandb_project,
                '--run_name', run_name,
                '--method', 'alps',
                '--sparsity', str(args.sp),
                '--gpu_util', str(args.gpu_util),
                '--tp_size', str(args.tp_size),
                '--flops', str(flops),
                '--profile', args.profile,
            ]
            if args.out_base:
                cmd += ['--out_base', args.out_base]
            if hub_model_id:
                cmd += ['--hub_model_id', hub_model_id, '--hub_url', hub_url]
            print(f'Running full eval: {" ".join(cmd)}')
            subprocess.run(cmd, check=True)
