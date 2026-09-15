"""MATH-500 end-to-end wall clock: dense vs an 80%-sparse model on MACKO.

Everything else in elsa/scripts/systems/ fixes the generation length so that a
kernel can be compared without the model's behaviour interfering. This one does
the opposite on purpose: it lets each model generate until EOS, because the
question here is what a user actually waits for.

That makes generated length part of the result, and it is the part most likely
to decide it. At 70% sparsity every MATH-500 rollout already hit the 8192 cap
with distinct-4 at 0.144 -- the model stops terminating and loops. A per-token
speedup of 1.26x is erased by generating 1.3x more tokens, so this reports
tokens and accuracy next to the clock. A "speedup" that comes from emitting
shorter garbage is not one, which is why accuracy is scored here rather than
assumed.

Both sides run the SAME inference path (HF generate). vLLM would be faster for
the dense side but MACKO has no vLLM integration -- it replaces nn.Linear
modules -- so using it for one side only would measure the serving stack, not
the model.

Usage: bench_math500_wallclock.py --dense <dir> --sparse <dir> [--n 100]
"""
import argparse, gc, json, re, time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import macko_spmv
_MULT = torch.ops.macko_spmv.multiply.default

LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class MackoLean(nn.Module):
    def __init__(self, compressed, out_features):
        super().__init__()
        c0, c1, c2, c3, c4 = compressed
        self.register_buffer("c0", c0, persistent=False)
        self.register_buffer("c1", c1, persistent=False)
        self.register_buffer("c2", c2, persistent=False)
        self.c3, self.c4 = c3, c4
        self.out_features = out_features

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] == 1:
            y = _MULT(self.c0, self.c1, self.c2, self.c3, self.c4, flat[0]).unsqueeze(0)
        else:
            y = torch.stack([_MULT(self.c0, self.c1, self.c2, self.c3, self.c4, flat[i])
                             for i in range(flat.shape[0])])
        return y.reshape(x.shape[:-1] + (self.out_features,))


def compress_model(model):
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or not any(t in name for t in LINEARS):
            continue
        w = mod.weight.data.to(torch.float16)
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1],
                MackoLean(macko_spmv.compress(w), w.shape[0]))
        del w
        n += 1
    gc.collect(); torch.cuda.empty_cache()
    return n


def last_boxed(s):
    """Pull the final \\boxed{...}, matching braces so nested ones survive."""
    i = s.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(s) and depth:
        depth += (s[j] == "{") - (s[j] == "}")
        j += 1
    return s[i + 7:j - 1].strip() if depth == 0 else None


def norm(x):
    if x is None:
        return None
    x = x.strip().replace(" ", "").replace("\\left", "").replace("\\right", "")
    x = re.sub(r"\\text\{([^}]*)\}", r"\1", x)
    x = re.sub(r"^\\\((.*)\\\)$", r"\1", x)
    return x.rstrip(".").rstrip("$").lstrip("$")


@torch.no_grad()
def run(model, tok, problems, max_new, tag):
    rows = []
    t_all = time.perf_counter()
    for k, p in enumerate(problems):
        msgs = [{"role": "user", "content": p["problem"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out[0, ids.shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True)
        n_tok = int(gen.shape[0])
        pred = norm(last_boxed(text))
        rows.append(dict(sec=dt, tokens=n_tok, tok_s=n_tok / dt,
                         truncated=n_tok >= max_new,
                         correct=(pred is not None and pred == norm(p["answer"]))))
        if (k + 1) % 10 == 0:
            el = time.perf_counter() - t_all
            print(f"  [{tag}] {k+1}/{len(problems)}  elapsed {el/60:.1f}min  "
                  f"mean {sum(r['tokens'] for r in rows)/len(rows):.0f} tok", flush=True)
    total = time.perf_counter() - t_all
    n = len(rows)
    return dict(
        total_sec=total,
        mean_sec=total / n,
        mean_tokens=sum(r["tokens"] for r in rows) / n,
        total_tokens=sum(r["tokens"] for r in rows),
        tok_s=sum(r["tokens"] for r in rows) / total,
        trunc_rate=sum(r["truncated"] for r in rows) / n,
        accuracy=sum(r["correct"] for r in rows) / n,
        per_problem=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=8192)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = [ds[i] for i in range(min(a.n, len(ds)))]
    print(f"torch {torch.__version__}  gpu={torch.cuda.get_device_name(0)}  "
          f"{len(problems)} problems, max_new={a.max_new}")

    tok = AutoTokenizer.from_pretrained(a.dense)
    res = {}

    for tag, path, sparse in (("dense", a.dense, False), ("macko_s80", a.sparse, True)):
        m = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
        if sparse:
            t0 = time.perf_counter()
            n = compress_model(m)
            print(f"compressed {n} Linears in {time.perf_counter()-t0:.0f}s", flush=True)
        print(f"--- {tag} ---", flush=True)
        res[tag] = run(m, tok, problems, a.max_new, tag)
        r = res[tag]
        print(f"  total {r['total_sec']/60:.1f}min  {r['mean_tokens']:.0f} tok/problem  "
              f"{r['tok_s']:.1f} tok/s  trunc {r['trunc_rate']:.2f}  acc {r['accuracy']:.3f}")
        del m; gc.collect(); torch.cuda.empty_cache()

    d, s = res["dense"], res["macko_s80"]
    print(f"\n{'':<22}{'dense':>12}{'MACKO s80':>12}{'ratio':>10}")
    print("-" * 56)
    for k, lbl in [("total_sec", "total wall-clock s"), ("mean_sec", "sec / problem"),
                   ("mean_tokens", "tokens / problem"), ("tok_s", "output tokens/s"),
                   ("trunc_rate", "truncation rate"), ("accuracy", "MATH-500 acc")]:
        x, y = d[k], s[k]
        r = x / y if y else float("nan")
        print(f"{lbl:<22}{x:12.3f}{y:12.3f}{r:9.3f}x")
    print("\nRead the first row against the fourth. tokens/s above 1 with total")
    print("wall-clock below 1 means the per-token win was spent on generating")
    print("more tokens -- which is what the accuracy and truncation rows are")
    print("there to explain.")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
