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

EVAL_FULL_SCRIPT = os.environ.get(
    "EVAL_FULL_SCRIPT", "/home1/doyoonkim/projects/elsa/scripts/eval_full.py")

DEV = torch.device('cuda')


# ── Data ─────────────────────────────────────────────────────────────────────

def get_ot_fw(nsamples, seed, seqlen, tokenizer, data_path, pack_short_docs=False):
    raw = load_dataset('json', data_files=data_path, split='train')
    random.seed(seed)
    np.random.seed(seed)

    # Tokenizing 40k+ long (8192+ token) docs one at a time in a plain Python
    # loop (the original approach here) measured at ~44s/300 docs (~100min
    # for the full corpus) on this box. Batched .map(num_proc=...) measured
    # ~9s/300 docs (~5x) -- same per-doc tokenizer() call, just spread across
    # worker processes instead of serialized in one. Not a threading issue
    # (torch.set_num_threads(1) made no difference); this is inherent
    # per-Python-call overhead x 40k docs, multiprocessing is what amortizes it.
    def _tok_batch(batch):
        return {'input_ids': [
            tokenizer(t).input_ids if t else [] for t in batch['text']
        ]}
    tokenized = raw.map(
        _tok_batch, batched=True, batch_size=32, num_proc=16,
        remove_columns=raw.column_names, desc='Tokenizing calibration docs',
    )

    if pack_short_docs:
        # NOTE: an earlier version of this branch concatenated multiple
        # DIFFERENT docs together (bos-separated) to fill out short ones --
        # measured to actively HURT self-gen calibration at s70 (math500
        # 2.6->0.0) versus the plain windowing path below. Root cause: the
        # concatenation buffer starts empty, so if the FIRST randomly-drawn
        # doc already exceeds seqlen (true for ~97% of self-gen v2 rows --
        # they're prompt+completion and the ORIGINAL prompt alone is often
        # long), the window is just that one doc's tokens [0:seqlen] with NO
        # random offset -- for a long prompt this can capture only the
        # prompt and cut off before the self-gen model's own completion ever
        # starts, silently defeating the entire point of self-gen
        # calibration. Fixed: keep each doc separate (no concatenation, per
        # explicit instruction), take a RANDOM offset window (matching the
        # plain path's proven-good behavior) for any doc >= seqlen, and drop
        # (not pad/concatenate) anything shorter -- this flag now only
        # differs from the plain path in intent/logging, not mechanics,
        # since essentially all rows here already clear seqlen; kept as a
        # distinct code path for future datasets with a higher short-doc
        # rate, where it'd matter more.
        all_tokens = []
        for row in tokenized:
            ids = row['input_ids']
            if len(ids) >= seqlen:
                all_tokens.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0))
        assert len(all_tokens) > 0, "No samples longer than seqlen"
        print(f'[pack_short_docs] {len(all_tokens)}/{len(tokenized)} docs >= seqlen={seqlen} '
              f'({len(tokenized) - len(all_tokens)} dropped as too short, no concatenation)')

        trainloader = []
        for _ in range(nsamples):
            src = random.choice(all_tokens)
            i = random.randint(0, src.shape[1] - seqlen)
            inp = src[:, i:i + seqlen]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    else:
        all_tokens = []
        for row in tokenized:
            ids = row['input_ids']
            if len(ids) >= seqlen:
                all_tokens.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0))
        assert len(all_tokens) > 0, "No samples longer than seqlen"

        trainloader = []
        for _ in range(nsamples):
            src = random.choice(all_tokens)
            i = random.randint(0, src.shape[1] - seqlen)
            inp = src[:, i:i + seqlen]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

    # Original approach here (join up to 500 raw docs into one string, then
    # tokenize that ~6.75M-char string in a single call) hung for 2h19m on
    # this box with 0% GPU util the whole time -- single massive-string
    # encode() calls appear to hit the same host-contention-sensitive
    # slowdown as the per-doc loop above did, just without multiprocessing
    # available to amortize it (it's one call, can't be split across
    # workers the same way). Reuse the per-doc token ids already computed
    # above instead of re-tokenizing anything -- just concatenate them in
    # dataset order until there's enough for the eval slice.
    eval_ids = []
    eval_len = 0
    target_len = 256 * seqlen
    for row in tokenized:
        ids = row['input_ids']
        if not ids:
            continue
        eval_ids.extend(ids)
        eval_len += len(ids)
        if eval_len >= target_len:
            break
    testenc_ids = torch.tensor(eval_ids, dtype=torch.long).unsqueeze(0)

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids

    return trainloader, TokenizerWrapper(testenc_ids[:, :256 * seqlen])


# ── Model ─────────────────────────────────────────────────────────────────────

def get_qwen3(model_path, seqlen=2048):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype='auto', trust_remote_code=True
    )
    model.seqlen = seqlen
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


# ── Corrected-target dense cache ────────────────────────────────────────────
#
# Standard ALPS reconstructs, at each layer, X_Q @ W_c -- the DENSE weight
# applied to the CURRENT (already partially pruned) trajectory. That's not
# actually "what the dense model would have produced here": the dense model
# never sees X_Q, it only ever sees its own (uncorrupted) trajectory X_P. This
# pre-pass runs the fully dense model ONCE (before any layer is touched) and
# caches each decoder layer's OWN input X_P_i to disk, so the main pruning
# loop can later re-run each (still-dense-at-that-point) layer a second time
# on X_P_i to get every submodule's true dense-reference output Y_P, used as
# the corrected reconstruction target instead of X_Q @ W_c. Layer-input-only
# (not per-submodule) keeps the cache to ~1GB/layer instead of exploding
# across every Linear inside the layer.
@torch.no_grad()
def cache_dense_layer_inputs(model, dataloader, dev, nsamples, cache_dir):
    print(f'Caching dense layer inputs to {cache_dir}...')
    os.makedirs(cache_dir, exist_ok=True)
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
    for i in range(len(layers)):
        torch.save(inps.cpu(), os.path.join(cache_dir, f'layer_{i}_X.pt'))
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = _layer_fwd(layer, inps[j].unsqueeze(0), cache)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    print('Dense layer input cache done.')
    return cache['attention_mask'], cache['position_ids'], cache['position_embeddings']


# ── Pruning ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def qwen3_sequential(model, dataloader, dev, args):
    print('Starting ALPS on Qwen3...')

    corrected_target = getattr(args, 'corrected_target', False)
    dense_cache_dir = getattr(args, 'dense_cache_dir', None)
    nsamples = args.nsamples

    if corrected_target:
        cache_dense_layer_inputs(model, dataloader, dev, nsamples, dense_cache_dir)

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

    skip_set = getattr(args, 'skip_layer_names', None) or {}

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        skip_here = skip_set.get(i, set())
        if skip_here:
            names_to_skip = list(full.keys()) if '*' in skip_here else [n for n in skip_here if n in full]
            for name in names_to_skip:
                print(f'  Layer {i} {name}: SKIPPED (kept dense)')
                del full[name]
        sequential = [list(full.keys())]

        Xp = None
        if corrected_target:
            Xp = torch.load(os.path.join(dense_cache_dir, f'layer_{i}_X.pt')).to(dev)

        scd = {}
        for names in sequential:
            subset = {n: full[n] for n in names}
            for name in subset:
                scd[name] = ALPS_prune(subset[name], nsamples=nsamples, seqlen=model.seqlen)

            # `mode['phase']` lets ONE set of hooks serve two different
            # forward passes through the SAME (still fully dense) layer: 'q'
            # accumulates H=X_Q^T X_Q as before (and stashes each submodule's
            # X_Q input for the immediately-following 'p' pass); 'p' -- only
            # run when corrected_target is on -- feeds the SAME layer the
            # dense-reference input X_P instead, and uses the resulting
            # (still-dense-weight) output as Y_P to accumulate the corrected
            # cross term X_Q^T Y_P.
            mode = {'phase': 'q'}
            current_xq = {}

            def make_hook(name):
                def tmp(_, inp, out):
                    if mode['phase'] == 'q':
                        scd[name].add_batch(inp[0].data, out.data)
                        if corrected_target:
                            current_xq[name] = inp[0].data
                    else:
                        scd[name].add_target_batch(current_xq[name], out.data)
                return tmp

            handles = [subset[name].register_forward_hook(make_hook(name)) for name in subset]
            for j in range(nsamples):
                mode['phase'] = 'q'
                _layer_fwd(layer, inps[j].unsqueeze(0), cache)
                if corrected_target:
                    mode['phase'] = 'p'
                    _layer_fwd(layer, Xp[j].unsqueeze(0), cache)
            for h in handles:
                h.remove()

            for name in subset:
                print(f'  Layer {i} {name}')
                scd[name].ALPS_admm(sp=args.sp, nm_n=args.nm_n, nm_m=args.nm_m, rho=args.rho,
                                     corrected_target=corrected_target)
                d1, d2 = scd[name].layer.weight.data.shape
                nnz = (scd[name].layer.weight.data.abs() > 0).sum().item()
                tot_params += d1 * d2
                tot_nnz += nnz
                scd[name].free()

        del Xp
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
    # lm_head(hidden_states) materializes a (1, seqlen, vocab) logits tensor --
    # at seqlen=8192 and vocab~152k that's ~2.5GB in bf16 BEFORE
    # CrossEntropyLoss's internal fp32 upcast/softmax roughly doubles it,
    # which OOM'd a 24GB RTX3090 once seqlen grew past ~4096 (this loop used
    # to run the whole sequence through lm_head + loss in one shot). Chunk
    # along the sequence dimension instead -- PPL is a sum of per-token NLL
    # regardless of how the tokens are grouped for the lm_head/loss call, so
    # this is numerically identical, just bounded peak memory.
    ppl_chunk = min(model.seqlen, 2048)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0).to(dev)
        if hasattr(model.model, 'norm') and model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        shift_labels_full = testenc[:, i * model.seqlen:(i + 1) * model.seqlen][:, 1:].to(dev)
        sample_nll = 0.0
        for c0 in range(0, model.seqlen - 1, ppl_chunk):
            c1 = min(c0 + ppl_chunk, model.seqlen - 1)
            lm_logits = model.lm_head(hidden_states[:, c0:c1 + 1, :])
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = shift_labels_full[:, c0:c1]
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            sample_nll += loss.float() * (c1 - c0)
            del lm_logits, shift_logits
        nlls.append(sample_nll)

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
    parser.add_argument('--pack_short_docs', action='store_true',
                         help='Pack docs shorter than seqlen instead of dropping them (fixes a bug where '
                              'self-gen calibration rows shorter than seqlen were silently discarded)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048,
                         help='Calibration window length in tokens (was hardcoded to 2048). Longer windows let '
                              'self-gen/CoT calibration docs (often 8k-16k+ tokens) actually cover mid/late '
                              'reasoning-trajectory activations instead of only ever seeing the first 2048 tokens.')
    parser.add_argument('--nm_n', type=int, default=0)
    parser.add_argument('--nm_m', type=int, default=0)
    parser.add_argument('--rho', type=float, default=300.0)
    parser.add_argument('--skip_layer_names', type=str, default='',
                         help="comma-separated layer_idx:proj_name pairs to keep dense, e.g. '35:mlp.up_proj' "
                              "or '35:mlp.up_proj,35:mlp.gate_proj'")
    parser.add_argument('--corrected_target', action='store_true',
                         help="Reconstruct each layer's OWN dense-reference output (computed by re-running that "
                              "still-dense layer on the dense model's own trajectory for the same calibration "
                              "tokens) instead of X_Q @ W_c (the dense weight applied to the current, possibly "
                              "already-pruned trajectory). Solver (ADMM/top-k/PCG) is unchanged -- only the "
                              "reconstruction target's linear term (YtX) is replaced. Doubles per-layer forward "
                              "cost (one pass on the corrupted trajectory for H, one pass on the cached dense "
                              "trajectory for the target) and needs ~1GB/layer of scratch disk for the dense "
                              "layer-input cache (see --dense_cache_dir).")
    parser.add_argument('--dense_cache_dir', type=str, default='',
                         help='Scratch dir for --corrected_target\'s per-layer dense-input cache (~1GB/layer). '
                              'Default: a subdir next to this script on real disk (NOT /tmp -- that\'s tmpfs on '
                              'this cluster and would eat into the SLURM job\'s --mem budget); auto-created, '
                              'not auto-deleted.')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--eval_full', action='store_true', help='Run full eval (PPL+zeroshot+lighteval) after pruning')
    parser.add_argument('--profile', type=str, default='quick', choices=['official', 'quick'],
                         help="lighteval profile passed through to eval_full.py: 'quick' (default, 8192 budget, "
                              "matches the rest of the dashboard) or 'official' (32768/38912 budget, incl. "
                              "AIME24/25)")
    parser.add_argument('--wandb_project', type=str, default='reasoning_qwen3_1.7b')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--gpu_util', type=float, default=0.9)
    parser.add_argument('--tp_size', type=int, default=1)
    parser.add_argument('--out_base', type=str, default='')
    parser.add_argument('--push_to_hub', action='store_true', help='Upload pruned model to HuggingFace Hub after saving')
    parser.add_argument('--hub_model_id', type=str, default=None, help='HF Hub repo id (e.g. username/model-name); auto-generated if not given')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = get_qwen3(args.model, seqlen=args.seqlen)
    model.eval()

    # One-shot calibration is forward-only: ~2*N*tokens (no backward pass),
    # unlike gradient fine-tuning's ~6*N*tokens.
    n_params = sum(p.numel() for p in model.parameters())
    n_tokens = args.nsamples * model.seqlen
    flops = 2 * n_params * n_tokens
    print(f'Calibration FLOPs: {flops:.3e} ({n_params} params x {n_tokens} tokens, forward-only)')

    dataloader, testenc = get_ot_fw(args.nsamples, args.seed, model.seqlen, tokenizer, args.data_path,
                                     pack_short_docs=args.pack_short_docs)

    skip_layer_names = {}
    if args.skip_layer_names:
        for pair in args.skip_layer_names.split(','):
            idx_str, name = pair.split(':')
            skip_layer_names.setdefault(int(idx_str), set()).add(name)
        print(f'Skipping (keeping dense): {skip_layer_names}')
    args.skip_layer_names = skip_layer_names

    if args.corrected_target and not args.dense_cache_dir:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.dense_cache_dir = os.path.join(script_dir, 'dense_cache_tmp', f'pid{os.getpid()}')

    tick = time.time()
    qwen3_sequential(model, dataloader, DEV, args)
    print(f'Pruning time: {time.time() - tick:.1f}s')

    if args.corrected_target and args.dense_cache_dir and os.path.isdir(args.dense_cache_dir):
        import shutil
        shutil.rmtree(args.dense_cache_dir, ignore_errors=True)
        print(f'Cleaned up dense cache dir: {args.dense_cache_dir}')

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
