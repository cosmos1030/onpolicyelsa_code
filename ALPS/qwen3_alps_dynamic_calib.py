"""
ALPS pruning with dynamic/EMA-style OT calibration data.

Standard ALPS calibration (qwen3_alps.py) freezes the OT80/FW20 calibration
set for the whole pruning run, built once from the DENSE model's own
distribution. This variant refreshes a fraction of the OT portion at every
single decoder-layer boundary, using completions sampled from the model
AS PARTIALLY PRUNED THROUGH THAT POINT -- so calibration data tracks the
model's evolving distribution instead of staying anchored to the original
dense one. Motivated by the KL-diagnostic finding that pruning damage
concentrates unevenly across layers (see ALPS/qwen3_alps_kldiag.py).

FineWeb-Edu (20%) stays completely fixed for the whole run (built once, same
as the static build). The OT (80%) portion is an evolving pool of exactly
n_ot fixed SLOTS (stable indices -- one slot's content survives across
layers unless it's the one being refreshed):
  - before layer 0: 50% original teacher-answer OT rows, 50% self-gen by
    the still-fully-dense model.
  - after layer i finishes pruning (i < n_layers-1): --refresh_ratio (default
    0.25) of the OT slots are replaced with fresh self-gen completions from
    the model as pruned through layer i; the rest keep their old content.

Same windowing rule as pure/static ALPS (qwen3_alps.py's get_ot_fw), for a
methodologically fair comparison -- ONLY text that is naturally >= seqlen
tokens is ever used, windowed via a random offset; anything shorter is
discarded outright, never padded/tiled/concatenated. Two earlier versions of
this file broke that parity: one tiled (repeated) short self-gen completions
to fill seqlen (an artificial repeated-content pattern no real training data
would ever have), the other padded them with a proper attention_mask (correct,
but not the same rule pure ALPS follows, so not a fair comparison). Since
self-gen completions are capped in length for cost reasons and often end
before reaching seqlen tokens (the model emits EOS), `generate_qualifying_texts`
oversamples -- generating extra candidates and keeping only the ones that
reach seqlen -- until it has exactly as many qualifying completions as
needed, instead of settling for a shorter one.

Sample sizing derived from ALPS's own --nsamples default (128, see
qwen3_alps.py): 20% FW = 26 samples (fixed), 80% OT = 102 samples.

Generation uses a resident vLLM engine (see build_vllm_engine /
sync_weights_to_vllm) synced in-place from the actively-pruned HF model's
current weights before each refresh, then put back to sleep -- the same
mechanism gmp_trainer.py's OPKD path uses to avoid an expensive engine
reload at every step. gen_max_new_tokens defaults to 8192, matching the
max_completion_length RAC's config_trace_ot3.yaml used to build the original
self-gen calibration set.

Cost note: a layer's calibration input depends on all prior (already-pruned)
layers, so naively recomputing every sample's input from embed_tokens at
every layer boundary costs O(n_layers^2) layer-forwards (~17x a static
ALPS run for 36 layers) -- prohibitively slow. This implementation instead
keeps the standard ALPS streaming scheme (`inps -> layer -> outs`, O(1)
layer-forward per sample per layer) for whichever slots are UNCHANGED since
the last refresh, and only pays the expensive "propagate from embed_tokens
through all already-pruned layers" cost for the (refresh_ratio-fraction)
slots that were just replaced -- since only their content is actually new.

Usage:
    python qwen3_alps_dynamic_calib.py <model> <sparsity> \
        --n_ot 102 --n_fw 26 --refresh_ratio 0.25 --seqlen 8192 \
        --gen_max_new_tokens 8192 \
        --save <out_dir>
"""

import argparse
import json
import os
import random
import sys
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

from modelutils import find_layers
from alps import ALPS_prune
from qwen3_alps import get_qwen3, _layer_fwd, qwen3_eval

sys.path.insert(0, '/home1/doyoonkim/projects/elsa/scripts')
from build_ot3_fineweb_dataset import _render_ot, _ROLE_MAP, _pack_fw

DEV = torch.device('cuda')


# ── Data: static FineWeb (fixed for the whole run) ───────────────────────────

def build_fw_fixed(n_fw, seed, tok, seqlen):
    ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    texts = []
    for row in ds:
        texts.append(row['text'])
        if len(texts) >= n_fw * 20:  # ample raw docs to pack from
            break
    records = _pack_fw(texts, tok, n_fw, seqlen)
    assert len(records) >= n_fw, f'only packed {len(records)}/{n_fw} FW windows'
    windows = []
    for r in records[:n_fw]:
        ids = tok(r['text'], add_special_tokens=False).input_ids[:seqlen]
        windows.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0))
    return windows


# ── Data: OT pool (teacher rows for init, self-gen prompts throughout) ──────

def load_teacher_ot_texts(n, seed, tok, min_tokens=64, seqlen=8192, num_proc=8):
    ds = load_dataset('open-thoughts/OpenThoughts3-1.2M', split='train')
    ds = ds.shuffle(seed=seed).select(range(min(n * 4, len(ds))))
    ds = ds.map(
        lambda b: _render_ot(b, tok, min_tokens, seqlen=seqlen, strip_think=False),
        batched=True, batch_size=64, num_proc=num_proc,
        remove_columns=[c for c in ds.column_names if c != 'conversations'],
        desc='Rendering teacher OT3 rows',
    )
    # text_to_window requires >= seqlen tokens (no padding/tiling path exists
    # anymore -- see module docstring), so filter against seqlen here, not
    # just min_tokens.
    ds = ds.filter(lambda ex: ex['text'] is not None and ex['n_tokens'] >= max(min_tokens, seqlen))
    texts = list(ds['text'])[:n]
    assert len(texts) >= n, f'only got {len(texts)}/{n} teacher OT rows >= {seqlen} tokens'
    return texts


def load_ot_prompt_pool(n, seed, tok, min_tokens=64):
    # A separate shuffle (seed+1) from the teacher pool so self-gen prompts
    # aren't just re-asking the same questions the teacher rows already cover.
    ds = load_dataset('open-thoughts/OpenThoughts3-1.2M', split='train')
    ds = ds.shuffle(seed=seed + 1).select(range(min(n * 4, len(ds))))
    prompts = []
    for convs in ds['conversations']:
        if isinstance(convs, str):
            convs = json.loads(convs)
        user_msg = next((c['value'] for c in convs if _ROLE_MAP.get(c['from'], c['from']) == 'user'), None)
        if user_msg and len(user_msg) > 0:
            prompts.append(f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n")
        if len(prompts) >= n:
            break
    assert len(prompts) >= n, f'only got {len(prompts)}/{n} self-gen prompts'
    return prompts


# ── Generation: resident vLLM engine, synced in-place from the HF model ────

def build_vllm_engine(model_path, max_model_len, gpu_mem_frac=0.4, enforce_eager=False):
    # Kept asleep (enable_sleep_mode=True offloads weights to CPU + drops KV
    # cache) except during an actual generation call -- avoids permanently
    # reserving GPU memory the HF model/ALPS Hessian collection also needs.
    # Same construction as gmp_trainer.py's single-GPU OPKD vLLM engine.
    os.environ.setdefault('VLLM_USE_V1', '0')
    from vllm import LLM
    engine = LLM(
        model_path, dtype='bfloat16', gpu_memory_utilization=gpu_mem_frac,
        trust_remote_code=True, max_model_len=max_model_len,
        enforce_eager=enforce_eager, enable_sleep_mode=True,
    )
    engine.sleep(1)
    return engine


def sync_weights_to_vllm(model, engine):
    # In-place weight copy into the resident vLLM engine's own parameter
    # tensors (matched by name) -- no reload/subprocess restart. Mirrors
    # gmp_trainer.py's _sync_opkd_weights_to_vllm (non-FSDP branch). Works
    # unchanged for sparsified weights: pruning only zeroes values, it never
    # changes shapes, so name-based matching still lines up.
    llm_engine = engine.llm_engine
    executor = llm_engine.engine_core.model_executor if hasattr(llm_engine, 'engine_core') else llm_engine.model_executor
    vllm_model = executor.driver_worker.model_runner.model
    vllm_state = {k: v for k, v in vllm_model.named_parameters()}
    for name, param in model.named_parameters():
        if name in vllm_state:
            vllm_state[name].data.copy_(param.data.to(vllm_state[name].dtype))


@torch.no_grad()
def generate_completions(model, engine, tok, prompts, max_new_tokens):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    engine.wake_up()
    sync_weights_to_vllm(model, engine)
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.6, top_p=0.95)
    tok_prompts = [TokensPrompt(prompt_token_ids=tok(p, add_special_tokens=False).input_ids) for p in prompts]
    outs = engine.generate(tok_prompts, sp)
    texts = []
    for p, o in zip(prompts, outs):
        comp = tok.decode(o.outputs[0].token_ids, skip_special_tokens=False)
        texts.append(p + comp)
    engine.sleep(1)
    return texts


def generate_qualifying_texts(model, engine, tok, prompt_pool, n_needed, seqlen, max_new_tokens,
                               oversample=3, max_rounds=6):
    # Same rule pure/static ALPS's get_ot_fw applies: only text that reaches
    # >= seqlen tokens is usable; anything shorter is discarded, never
    # padded/tiled. Self-gen completions are capped in length for cost
    # reasons and often end (EOS) before reaching seqlen, so this oversamples
    # -- generating extra candidates each round and keeping only the ones
    # that qualify -- until exactly n_needed qualifying completions exist.
    qualifying = []
    rounds = 0
    while len(qualifying) < n_needed and rounds < max_rounds:
        batch_n = (n_needed - len(qualifying)) * oversample
        prompts = [random.choice(prompt_pool) for _ in range(batch_n)]
        texts = generate_completions(model, engine, tok, prompts, max_new_tokens=max_new_tokens)
        for t in texts:
            ids = tok(t, add_special_tokens=False).input_ids
            if len(ids) >= seqlen:
                qualifying.append(t)
                if len(qualifying) >= n_needed:
                    break
        rounds += 1
        print(f'  [generate_qualifying_texts] round {rounds}: {len(qualifying)}/{n_needed} qualifying '
              f'(>= {seqlen} tok)', flush=True)
    assert len(qualifying) >= n_needed, (
        f'only got {len(qualifying)}/{n_needed} qualifying (>= {seqlen} tok) self-gen completions '
        f'after {rounds} rounds -- try raising --gen_max_new_tokens or --oversample')
    return qualifying[:n_needed]


def text_to_window(text, tok, seqlen):
    # text must already be >= seqlen tokens (caller's responsibility --
    # load_teacher_ot_texts/build_fw_fixed/generate_qualifying_texts all
    # guarantee this). Random-offset window, matching static ALPS exactly.
    bos = tok.bos_token or ''
    eos = tok.eos_token or ''
    ids = tok(bos + text + eos, add_special_tokens=False).input_ids
    assert len(ids) >= seqlen, f'text_to_window got {len(ids)} < seqlen={seqlen} tokens'
    i = random.randint(0, len(ids) - seqlen)
    return torch.tensor(ids[i:i + seqlen], dtype=torch.long).unsqueeze(0)


# ── Per-layer input capture (full-model forward, stop at target layer) ─────

class _StopFwd(Exception):
    pass


@torch.no_grad()
def capture_inps_for_layer(model, layer_idx, windows, dev):
    # Forwards each window from embed_tokens through layers[0:layer_idx]
    # (all already pruned in place up to this point) and captures the
    # hidden state fed INTO layers[layer_idx]. Used for (a) the cheap
    # initial capture at layer_idx=0 (no layer compute at all) and (b) the
    # deliberately expensive "catch-up" propagation for freshly-generated
    # calibration slots that never existed before this layer boundary. Every
    # window is exactly seqlen tokens (no padding), so attention_mask/
    # position_ids/position_embeddings are identical across samples and can
    # be captured once and shared -- same as static ALPS's Catcher.
    layer = model.model.layers[layer_idx]
    n = len(windows)
    dtype = next(model.parameters()).dtype
    hidden = model.config.hidden_size
    seqlen = windows[0].shape[1]
    inps = torch.zeros((n, seqlen, hidden), dtype=dtype, device=dev)
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
    for j, w in enumerate(windows):
        state['i'] = j
        try:
            model(w.to(dev))
        except _StopFwd:
            pass
    h.remove()
    return inps, cache


@torch.no_grad()
def prune_layer(layer, inps, cache, nsamples, args):
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

    tot_params, tot_nnz = 0, 0
    for name in full:
        print(f'  {name}', flush=True)
        scd[name].ALPS_admm(sp=args.sp, nm_n=args.nm_n, nm_m=args.nm_m, rho=args.rho)
        d1, d2 = scd[name].layer.weight.data.shape
        nnz = (scd[name].layer.weight.data.abs() > 0).sum().item()
        tot_params += d1 * d2
        tot_nnz += nnz
        scd[name].free()
    return tot_params, tot_nnz


# ── Main pruning loop ───────────────────────────────────────────────────────

@torch.no_grad()
def qwen3_sequential_dynamic(model, engine, tok, dev, args, ot_pool_texts, ot_prompt_pool, fw_windows, log_path):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.to(dev)
    layers = model.model.layers
    n_layers = min(len(layers), args.max_layers) if getattr(args, 'max_layers', 0) else len(layers)

    n_ot = len(ot_pool_texts)
    n_fw = len(fw_windows)
    nsamples = n_ot + n_fw
    refresh_n = max(1, round(n_ot * args.refresh_ratio))
    seqlen = args.seqlen

    # Stable-index arrays: slots [0, n_ot) are OT (mutable identity,
    # refreshed fractionally each layer), slots [n_ot, nsamples) are FW
    # (fixed forever).
    windows = [text_to_window(t, tok, seqlen) for t in ot_pool_texts] + list(fw_windows)

    # Initial capture: input to layer 0 == embed_tokens output, no decoder
    # layer compute involved -- cheap regardless of which samples these are.
    inps, cache = capture_inps_for_layer(model, 0, windows, dev)

    tot_params, tot_nnz = 0, 0
    log_f = open(log_path, 'a')

    for i in range(n_layers):
        t0 = time.time()
        layer = layers[i]
        p, nnz = prune_layer(layer, inps, cache, nsamples, args)
        tot_params += p
        tot_nnz += nnz
        t_prune = time.time() - t0

        # Base pass: input-to-layer-(i+1) for every CURRENT slot, one
        # single-layer forward each -- same O(nsamples) cost per layer as
        # standard static ALPS, valid for whichever content occupies each
        # slot right now.
        outs = torch.zeros_like(inps)
        for j in range(nsamples):
            outs[j] = _layer_fwd(layer, inps[j].unsqueeze(0), cache)
        del inps
        torch.cuda.empty_cache()

        t1 = time.time()
        if i < n_layers - 1:
            refresh_idx = random.sample(range(n_ot), refresh_n)
            fresh_texts = generate_qualifying_texts(
                model, engine, tok, ot_prompt_pool, refresh_n, seqlen, args.gen_max_new_tokens,
                oversample=args.oversample,
            )
            refreshed_windows = []
            for k, idx in enumerate(refresh_idx):
                ot_pool_texts[idx] = fresh_texts[k]
                w = text_to_window(fresh_texts[k], tok, seqlen)
                windows[idx] = w
                refreshed_windows.append(w)

            # Expensive catch-up: ONLY the just-replaced slots need their
            # input-to-layer-(i+1) propagated from embed_tokens through all
            # i+1 already-pruned layers (their old outs[] entry is stale --
            # it reflects the text that used to be in that slot).
            catchup_inps, _ = capture_inps_for_layer(model, i + 1, refreshed_windows, dev)
            for k, idx in enumerate(refresh_idx):
                outs[idx] = catchup_inps[k]
            del catchup_inps
            torch.cuda.empty_cache()
        t_gen = time.time() - t1

        inps = outs
        rec = {'layer': i, 'sparsity_running': 1 - tot_nnz / tot_params,
               'prune_time_s': t_prune, 'refresh_time_s': t_gen}
        print(f'Layer {i}: pruned in {t_prune:.1f}s, refresh in {t_gen:.1f}s, '
              f'running sparsity={rec["sparsity_running"]:.4f}', flush=True)
        log_f.write(json.dumps(rec) + '\n')
        log_f.flush()

    log_f.close()
    model.config.use_cache = use_cache
    actual_sp = 1 - tot_nnz / tot_params
    print(f'ALPS dynamic-calib pruning done. Actual sparsity: {actual_sp:.4f}')
    return actual_sp, ot_pool_texts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('sp', type=float, help='Sparsity level')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--seqlen', type=int, default=8192)
    parser.add_argument('--n_ot', type=int, default=102, help='OT calibration slots (80% of nsamples=128)')
    parser.add_argument('--n_fw', type=int, default=26, help='Fixed FW calibration windows (20% of nsamples=128)')
    parser.add_argument('--refresh_ratio', type=float, default=0.25,
                         help='fraction of OT slots refreshed via self-gen at each layer boundary')
    parser.add_argument('--gen_max_new_tokens', type=int, default=8192,
                         help='matches RAC config_trace_ot3.yaml max_completion_length used for the original self-gen build')
    parser.add_argument('--oversample', type=int, default=3,
                         help='generate this many x candidates per needed slot when filtering to >= seqlen completions')
    parser.add_argument('--vllm_gpu_mem', type=float, default=0.4,
                         help='vLLM gpu_memory_utilization for the resident generation engine (kept asleep between refreshes)')
    parser.add_argument('--vllm_max_prompt_len', type=int, default=1024)
    parser.add_argument('--vllm_enforce_eager', action='store_true')
    parser.add_argument('--max_layers', type=int, default=0,
                         help='stop after this many decoder layers (0 = all); for smoke-testing the pipeline')
    parser.add_argument('--nm_n', type=int, default=0)
    parser.add_argument('--nm_m', type=int, default=0)
    parser.add_argument('--rho', type=float, default=300.0)
    parser.add_argument('--min_tokens', type=int, default=64)
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--out', type=str, default='dynamic_calib_log.jsonl')
    parser.add_argument('--push_to_hub', action='store_true')
    parser.add_argument('--hub_model_id', type=str, default=None)
    parser.add_argument('--skip_eval', action='store_true',
                         help='skip the final PPL eval (its 40k-doc tokenization pass is slow); for smoke-testing')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = get_qwen3(args.model)
    model.seqlen = args.seqlen
    model.eval()

    print(f'Building fixed FW pool ({args.n_fw} windows @ seqlen={args.seqlen})...', flush=True)
    fw_windows = build_fw_fixed(args.n_fw, args.seed, tokenizer, args.seqlen)

    half = args.n_ot // 2
    print(f'Loading {half} teacher OT rows and self-gen prompt pool...', flush=True)
    teacher_texts = load_teacher_ot_texts(half, args.seed, tokenizer, args.min_tokens, args.seqlen)
    ot_prompt_pool = load_ot_prompt_pool(max(args.n_ot * 4, 256), args.seed, tokenizer, args.min_tokens)

    print(f'Building resident vLLM engine (gpu_mem={args.vllm_gpu_mem}, '
          f'max_model_len={args.vllm_max_prompt_len + args.gen_max_new_tokens})...', flush=True)
    engine = build_vllm_engine(
        args.model, args.vllm_max_prompt_len + args.gen_max_new_tokens,
        gpu_mem_frac=args.vllm_gpu_mem, enforce_eager=args.vllm_enforce_eager,
    )

    print(f'Self-gen initial {args.n_ot - half} qualifying (>= seqlen) completions with the DENSE model...', flush=True)
    model.to(DEV)
    dense_selfgen_texts = generate_qualifying_texts(
        model, engine, tokenizer, ot_prompt_pool, args.n_ot - half, args.seqlen, args.gen_max_new_tokens,
        oversample=args.oversample,
    )
    ot_pool_texts = teacher_texts + dense_selfgen_texts

    tick = time.time()
    actual_sp, final_ot_pool = qwen3_sequential_dynamic(
        model, engine, tokenizer, DEV, args, ot_pool_texts, ot_prompt_pool, fw_windows, args.out,
    )
    print(f'Pruning time: {time.time() - tick:.1f}s')

    if not args.skip_eval:
        # Eval PPL on a fixed held-out slice (same corpus used elsewhere),
        # independent of the dynamic calibration pool.
        from qwen3_alps import get_ot_fw
        _, testenc = get_ot_fw(
            4, args.seed, args.seqlen, tokenizer,
            '/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl',
        )
        ppl = qwen3_eval(model, testenc, DEV)
        print(f'Final PPL: {ppl:.4f}')

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)
        print(f'Saved to {args.save}')

    if args.push_to_hub and args.save:
        try:
            from huggingface_hub import HfApi
            for _env in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
                os.environ.pop(_env, None)
            hub_model_id = args.hub_model_id or f"cosmos1030/alps-dyncalib-s{int(args.sp*100)}pct"
            api = HfApi()
            api.create_repo(repo_id=hub_model_id, exist_ok=True)
            api.upload_folder(folder_path=args.save, repo_id=hub_model_id,
                               commit_message=f'ALPS dynamic-calib pruned: sparsity={args.sp}')
            print(f'Uploaded to https://huggingface.co/{hub_model_id}')
        except Exception as e:
            print(f'WARNING: push_to_hub failed ({e}); continuing without upload.')
