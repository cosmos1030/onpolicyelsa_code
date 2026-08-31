import argparse
import os
import random
import subprocess
import sys
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from pruning_utils import find_layers, SparseGPT_LlaMA
from quant import Quantizer

EVAL_FULL_SCRIPT = "/home1/doyoonkim/projects/elsa/scripts/eval_full.py"


# ── Data ─────────────────────────────────────────────────────────────────────

def get_ot_fw(nsamples, seed, seqlen, tokenizer, data_path):
    raw = load_dataset('json', data_files=data_path, split='train')
    random.seed(seed)

    # Tokenizing docs one at a time in a plain Python loop (the original
    # approach here) measured at ~44s/300 docs (~100min for a 40k-doc
    # corpus) -- same per-doc tokenizer() call, just spread across worker
    # processes via batched .map(num_proc=...) instead of serialized in one
    # amortizes it ~5x. Ported from ALPS/qwen3_alps.py's get_ot_fw, which hit
    # and fixed the identical bottleneck on this same calibration data.
    def _tok_batch(batch):
        return {'input_ids': [
            tokenizer(t).input_ids if t else [] for t in batch['text']
        ]}
    tokenized = raw.map(
        _tok_batch, batched=True, batch_size=32, num_proc=16,
        remove_columns=raw.column_names, desc='Tokenizing calibration docs',
    )

    all_tokens = []
    for row in tokenized:
        ids = row['input_ids']
        if len(ids) >= seqlen:
            all_tokens.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0))
    assert len(all_tokens) > 0, "No samples longer than seqlen found"

    trainloader = []
    for _ in range(nsamples):
        src = random.choice(all_tokens)
        i = random.randint(0, src.shape[1] - seqlen)
        inp = src[:, i:i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # testenc: concatenate a chunk for PPL eval. Original approach here
    # (join up to 500 raw docs into one string, then tokenize that single
    # multi-million-char string in one call) hung for 2h19m with 0% GPU util
    # on the same host-contention-sensitive pattern as the per-doc loop
    # above, just without multiprocessing available to amortize it. Reuse
    # the per-doc token ids already computed above instead of re-tokenizing
    # anything -- concatenate them in dataset order until there's enough.
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
    """Return a Catcher class that captures inputs + Qwen3 kwargs."""
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


def _layer_forward(layer, inp, attention_mask, position_ids, position_embeddings):
    kwargs = {}
    if attention_mask is not None:
        kwargs['attention_mask'] = attention_mask
    if position_ids is not None:
        kwargs['position_ids'] = position_ids
    if position_embeddings is not None:
        kwargs['position_embeddings'] = position_embeddings
    return layer(inp, **kwargs)[0]


# ── Pruning ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def qwen3_sparsellm(model, dataloader, dev, args):
    print("Starting SparseLLM on Qwen3...")

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
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'position_embeddings': None}

    Catcher = _make_catcher(inps, cache, args.nsamples)
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
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    print("Calibration data ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        sequential = [list(full.keys())] if not args.true_sequential else [
            ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
            ["self_attn.o_proj"],
            ["mlp.up_proj", "mlp.gate_proj"],
            ["mlp.down_proj"],
        ]

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}

            gpts = {}
            for name in subset:
                if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                    continue
                gpts[name] = SparseGPT_LlaMA(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data, name)
                return tmp

            handles = [subset[name].register_forward_hook(add_batch(name)) for name in gpts]
            for j in range(args.nsamples):
                _layer_forward(layer, inps[j].unsqueeze(0), attention_mask, position_ids, position_embeddings)
            for h in handles:
                h.remove()

            for name in gpts:
                print(f"Layer {i}, {name}")
                gpts[name].fasterprune(
                    args.sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )
                gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = _layer_forward(layer, inps[j].unsqueeze(0), attention_mask, position_ids, position_embeddings)

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    print("SparseLLM pruning done.")


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def qwen3_eval(model, testenc, dev, args):
    print("Evaluating...")
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
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = _layer_forward(layer, inps[j].unsqueeze(0), attention_mask, position_ids, position_embeddings)
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
    print(f"PPL: {ppl.item():.4f}")
    model.config.use_cache = use_cache
    return ppl.item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True, help="Path to ot+fw JSONL file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--gmp", action="store_true")
    parser.add_argument("--wbits", type=int, default=16)
    parser.add_argument("--minlayer", type=int, default=-1)
    parser.add_argument("--maxlayer", type=int, default=1000)
    parser.add_argument("--prune_only", type=str, default="")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--true-sequential", action="store_true")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--eval_full", action="store_true", help="Run full eval (PPL+zeroshot+lighteval) after pruning")
    parser.add_argument("--wandb_project", type=str, default="reasoning_qwen3_1.7b")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--gpu_util", type=float, default=0.9)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--out_base", type=str, default="")
    parser.add_argument('--profile', type=str, default='quick', choices=['official', 'quick'],
                         help="lighteval profile passed through to eval_full.py: 'quick' (default, 8192 budget, "
                              "matches the rest of the dashboard) or 'official' (32768 budget). eval_full.py's own "
                              "default is now also 'quick' -- this flag's default just keeps the two in sync "
                              "explicitly rather than relying on that.")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload pruned model to HuggingFace Hub after saving")
    parser.add_argument("--hub_model_id", type=str, default=None, help="HF Hub repo id (e.g. username/model-name); auto-generated if not given")
    args = parser.parse_args()

    MODEL = args.model
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = get_qwen3(MODEL)
    model.eval()

    # One-shot calibration is forward-only: ~2*N*tokens (no backward pass),
    # unlike gradient fine-tuning's ~6*N*tokens.
    n_params = sum(p.numel() for p in model.parameters())
    n_tokens = args.nsamples * args.seqlen
    flops = 2 * n_params * n_tokens
    print(f"Calibration FLOPs: {flops:.3e} ({n_params} params x {n_tokens} tokens, forward-only)")

    dataloader, testenc = get_ot_fw(args.nsamples, args.seed, args.seqlen, tokenizer, args.data_path)

    dev = torch.device('cuda')
    if args.sparsity or args.prunen:
        qwen3_sparsellm(model, dataloader, dev, args)

    ppl = qwen3_eval(model, testenc, dev, args)
    print(f"Final PPL on ot_fw: {ppl:.4f}")

    if args.save:
        model.save_pretrained(args.save)
        tokenizer.save_pretrained(args.save)
        print(f"Saved to {args.save}")

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
                hub_model_id = f"cosmos1030/sparsellm-s{int(args.sparsity * 100)}pct_{_now}"
            print(f"Uploading model to HuggingFace Hub: {hub_model_id}")
            api = HfApi()
            api.create_repo(repo_id=hub_model_id, exist_ok=True)
            api.upload_folder(
                folder_path=args.save,
                repo_id=hub_model_id,
                commit_message=f"SparseLLM pruned: sparsity={args.sparsity}",
            )
            hub_url = f"https://huggingface.co/{hub_model_id}"
            print(f"Uploaded to {hub_url}")
        except Exception as e:
            print(f"WARNING: push_to_hub failed ({e}); continuing without upload.")
            hub_model_id = None
            hub_url = None

    if args.eval_full:
        if not args.save:
            print("WARNING: --eval_full requires --save; skipping full eval")
        else:
            # Release GPU memory before launching vLLM in eval_full subprocess
            import gc
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print("GPU memory released before eval_full subprocess")

            run_name = args.run_name or f"sparsellm_s{int(args.sparsity * 100)}pct"
            cmd = [
                sys.executable, EVAL_FULL_SCRIPT,
                "--model_path", args.save,
                "--wandb_project", args.wandb_project,
                "--run_name", run_name,
                "--method", "sparsellm",
                "--sparsity", str(args.sparsity),
                "--gpu_util", str(args.gpu_util),
                "--tp_size", str(args.tp_size),
                "--flops", str(flops),
                "--profile", args.profile,
            ]
            if args.out_base:
                cmd += ["--out_base", args.out_base]
            if hub_model_id:
                cmd += ["--hub_model_id", hub_model_id, "--hub_url", hub_url]
            print(f"Running full eval: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
