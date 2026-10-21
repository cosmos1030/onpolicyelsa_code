"""Batched projection sampling (Karan, Chen, Du, "Finetuning with Sampling", arXiv 2610.02140;
github.com/aakaran/finetuning-with-sampling, boost_math.py) for Qwen3 thinking-mode traces.

Same algorithm as proj_samp_vLLM, but every vLLM call serves a whole batch of problems at once
instead of one problem at a time (the original makes ~block_num*(1+2*mcmc_steps) sequential calls
per problem):

  for each block of `jump` tokens:
    propose: continue the current response with the model shown the problem AND its reference
             solution (SEQ_PROMPT_MATH), temperature 0.6, repetition penalty 1.1
    score:   mean token log-prob of the response under the model shown the problem ONLY
    mcmc_steps times: cut at a random position, re-propose the suffix (same length), keep it iff
             the problem-only score improves (greedy, as in the original)
  stop a problem at EOS

Output jsonl rows {"text": chat-formatted user+assistant turn, ...} -- the same "text" format as
the OT3 training files, so the SCOUT trainer reads it unchanged as its NTP data.
"""
import argparse
import json
import os
import random
import time

import pandas as pd

SEQ_PROMPT_MATH = """You are an expert mathematician given a math problem, its solution, and an initial, partial response. Carefully study the solution,
identifying what reasoning or steps are already provided, and then continue the partial response. Ensure your response is logically consistent with the solution and leads to a complete and correct final answer.

Task:

{PROBLEM}

This is an example for a solution to the problem:

{SOLUTION}

Here is a partial response:

{RESPONSE}

Starting with the partial response, continue the response in your own words, including the thinking process. Ensure the final answer exactly matches that of the provided solution, and present the final answer within \\boxed{{}}.
"""
# Domain-neutral version of SEQ_PROMPT_MATH for mixed corpora (OT3: math, code, science): same
# structure and instructions, minus "mathematician" and the \\boxed requirement.
SEQ_PROMPT_GENERIC = """You are an expert given a problem, its solution, and an initial, partial response. Carefully study the solution,
identifying what reasoning or steps are already provided, and then continue the partial response. Ensure your response is logically consistent with the solution and leads to a complete and correct final answer.

Task: 

{PROBLEM}

This is an example for a solution to the problem: 

{SOLUTION}

Here is a partial response: 

{RESPONSE}

Starting with the partial response, continue the response in your own words, including the thinking process. Ensure the final answer matches that of the provided solution.
Do not mention, quote, or refer to the provided solution (or the fact that a solution was given) anywhere in your response; write as if you were solving the problem on your own.
"""
import re
# a response that talks about the hint it was shown ("the provided solution says ...") -- the trainee
# never sees that hint, so such text teaches it to refer to a solution that is not there
LEAK = re.compile(r'(provided|given|example|reference|sample) solution|the solution (says|mentions|shows|uses|given|provided)'
                  r'|solution (provided|given) (above|earlier)', re.I)
MATH_PROMPT = "You are an expert mathematician. Can you solve the following math problem? "
MATH_COT = "\n\nExplain your reasoning, thinking step by step, and return your final answer within \\boxed{}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True, help='parquet with problem/solution columns')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=-1)
    ap.add_argument('--batch', type=int, default=256, help='problems processed together')
    ap.add_argument('--max_new_tokens', type=int, default=8192)
    ap.add_argument('--block_num', type=int, default=32)
    ap.add_argument('--mcmc_steps', type=int, default=10)
    ap.add_argument('--enable_thinking', type=int, default=1)
    ap.add_argument('--gpu_mem', type=float, default=0.6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prompt_style', default='math', choices=['math', 'raw'],
                    help="math: MATH_PROMPT+problem+MATH_COT and SEQ_PROMPT_MATH (original); raw: problem as-is and SEQ_PROMPT_GENERIC (OT3)")
    ap.add_argument('--reject_leak', type=int, default=0,
                    help='re-propose a block (up to this many times) whose text refers to the hint solution; '
                         'an MCMC candidate that still does is rejected')
    ap.add_argument('--idx_file', default='', help='json list of row indices to process instead of --start/--end')
    ap.add_argument('--plain', action='store_true',
                    help='control: one ordinary sample per problem from the problem alone (no solution hint, no MCMC)')
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    random.seed(args.seed)
    df = pd.read_parquet(args.data)
    end = len(df) if args.end < 0 else min(args.end, len(df))
    rows = list(range(args.start, end))
    if args.idx_file:
        rows = json.load(open(args.idx_file))
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)['idx'] for l in f if l.strip()}
    rows = [i for i in rows if i not in done]
    print(f'{len(rows)} problems to do ({len(done)} already in {args.out})', flush=True)

    assert args.max_new_tokens % args.block_num == 0
    jump = args.max_new_tokens // args.block_num
    # vLLM only generates. Scoring needs prompt log-probs, which vLLM 0.10 (V0) computes by
    # materialising several [tokens x 152k-vocab] tensors (OOM on 48GB at 7k tokens) and cannot
    # combine with chunked prefill + prefix caching -- so scoring runs on an HF copy of the model
    # instead, a few hundred positions of logits at a time.
    # the proposal prompt carries the partial response twice (as text inside SEQ_PROMPT and as the raw
    # continuation prefix, as in the original), so it can reach ~2x the response length plus the
    # problem and reference solution
    max_len = 2 * args.max_new_tokens + 3072
    llm = LLM(model=args.model, gpu_memory_utilization=args.gpu_mem, seed=args.seed,
              max_model_len=max_len, enable_prefix_caching=True)
    import torch
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()
    tok = llm.get_tokenizer()
    eos = tok.eos_token_id
    im_end = tok.convert_tokens_to_ids('<|im_end|>')
    stop_ids = sorted({eos, im_end})

    def chat(text):
        return tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=bool(args.enable_thinking))

    def propose(items):
        """items: (question, solution, prefix_tokens, n_new). Returns new tokens per item
        (empty for an item whose prompt would not fit -- the caller then keeps what it has)."""
        prompts, sps, keep = [], [], []
        for j, (q, sol, cur, n) in enumerate(items):
            ctx = chat(SEQ.format(PROBLEM=q, SOLUTION=sol,
                                              RESPONSE=tok.decode(cur, skip_special_tokens=True)))
            ids = tok.encode(ctx, add_special_tokens=False) + cur
            n = max(1, min(n, max_len - len(ids) - 1))
            if len(ids) + 1 >= max_len:
                continue
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            sps.append(SamplingParams(max_tokens=n, temperature=0.6, repetition_penalty=1.1,
                                      stop_token_ids=stop_ids, seed=random.randrange(1 << 30)))
            keep.append(j)
        res = [[] for _ in items]
        if prompts:
            for j, o in zip(keep, llm.generate(prompts, sps, use_tqdm=False)):
                res[j] = list(o.outputs[0].token_ids)
        return res

    def leaks(prefix, new):
        # the new block plus a little of what precedes it, so a phrase split across the boundary is caught
        return bool(LEAK.search(tok.decode(prefix[-32:] + new, skip_special_tokens=True)))

    def propose_clean(items, final):
        """propose(), re-drawing (up to --reject_leak times) every block whose text mentions the hint.
        final=True (the forward block): a block still leaking after the retries is kept, there is nothing
        to fall back to; final=False (an MCMC candidate): it is dropped, i.e. the candidate is rejected."""
        res = propose(items)
        if not args.reject_leak:
            return res
        for _ in range(args.reject_leak):
            bad = [j for j, (it, r) in enumerate(zip(items, res)) if r and leaks(it[2], r)]
            if not bad:
                break
            for j, r in zip(bad, propose([items[j] for j in bad])):
                res[j] = r
        if not final:
            res = [[] if r and leaks(it[2], r) else r for it, r in zip(items, res)]
        return res

    @torch.no_grad()
    def score(items, max_tokens_per_batch=24576):
        """items: (question, response_tokens). Mean log-prob of the response given the question only.
        Sequences are length-sorted and right-padded into batches of at most max_tokens_per_batch
        tokens; logits are formed a few hundred positions at a time and only the target tokens'
        log-probs are kept, so the 152k-vocab never materialises for a whole sequence."""
        enc = []
        for j, (q, resp) in enumerate(items):
            ctx = tok.encode(chat(q), add_special_tokens=False)
            enc.append((j, ctx, resp))
        res = [float('-inf')] * len(items)
        enc = [e for e in enc if e[2]]
        enc.sort(key=lambda e: len(e[1]) + len(e[2]))
        i = 0
        while i < len(enc):
            L = len(enc[i][1]) + len(enc[i][2])
            n = 1
            while i + n < len(enc) and (n + 1) * (len(enc[i + n][1]) + len(enc[i + n][2])) <= max_tokens_per_batch:
                n += 1
            grp = enc[i:i + n]
            Lmax = max(len(c) + len(r) for _, c, r in grp)
            ids = torch.full((len(grp), Lmax), tok.pad_token_id or 0, dtype=torch.long, device='cuda')
            att = torch.zeros((len(grp), Lmax), dtype=torch.long, device='cuda')
            for b, (_, c, r) in enumerate(grp):
                seq = torch.tensor(c + r, device='cuda')
                ids[b, :len(seq)] = seq
                att[b, :len(seq)] = 1
            h = hf.model(input_ids=ids, attention_mask=att).last_hidden_state
            for b, (j, c, r) in enumerate(grp):
                hs = h[b, len(c) - 1:len(c) + len(r) - 1]
                tgt = ids[b, len(c):len(c) + len(r)]
                tot = 0.0
                for k in range(0, hs.shape[0], 512):
                    lp = torch.log_softmax(hf.lm_head(hs[k:k + 512]).float(), -1)
                    tot += lp.gather(1, tgt[k:k + 512, None]).sum().item()
                res[j] = tot / len(r)
            del h
            i += n
        return res

    SEQ = SEQ_PROMPT_MATH if args.prompt_style == 'math' else SEQ_PROMPT_GENERIC

    if args.plain:
        # control arm: ordinary self-generation, same prompts, same token budget, no hint, no MCMC
        sp = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.6, top_p=0.95, top_k=20,
                            stop_token_ids=stop_ids, seed=args.seed)
        for b0 in range(0, len(rows), args.batch):
            idxs = rows[b0:b0 + args.batch]
            Q = [(MATH_PROMPT + df['problem'][i] + MATH_COT) if args.prompt_style == 'math' else df['problem'][i]
                 for i in idxs]
            outs = llm.generate([TokensPrompt(prompt_token_ids=tok.encode(chat(q), add_special_tokens=False)) for q in Q],
                                sp, use_tqdm=False)
            with open(args.out, 'a') as f:
                for q, i, o in zip(Q, idxs, outs):
                    g = list(o.outputs[0].token_ids)
                    fin = bool(g) and g[-1] in stop_ids or o.outputs[0].finish_reason == 'stop'
                    g = [t for t in g if t not in stop_ids]
                    resp = tok.decode(g, skip_special_tokens=False)
                    # a response cut at the token budget gets no end-of-turn token: the model never ended it
                    end = '<|im_end|>\n' if fin else ''
                    f.write(json.dumps({'idx': i, 'text': f'<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{resp}{end}',
                                        'finished': bool(fin), 'n_tokens': len(g), 'solution': df['solution'][i]}) + '\n')
            print(f'plain batch {b0}: {len(idxs)} done', flush=True)
        return

    for b0 in range(0, len(rows), args.batch):
        idxs = rows[b0:b0 + args.batch]
        t0 = time.time()
        Q = [(MATH_PROMPT + df['problem'][i] + MATH_COT) if args.prompt_style == 'math' else df['problem'][i]
             for i in idxs]
        S = [df['solution'][i] for i in idxs]
        gen = [[] for _ in idxs]
        cur = [float('-inf')] * len(idxs)
        alive = [True] * len(idxs)
        acc = att = 0
        for blk in range(args.block_num):
            act = [k for k in range(len(idxs)) if alive[k]]
            if not act:
                break
            new = propose_clean([(Q[k], S[k], gen[k], jump) for k in act], final=True)
            for k, n in zip(act, new):
                gen[k] = gen[k] + n
            for k, s in zip(act, score([(Q[k], gen[k]) for k in act])):
                cur[k] = s
            for _ in range(args.mcmc_steps):
                cand = [k for k in act if len(gen[k]) > 0]
                cut = {k: random.randint(0, len(gen[k]) - 1) for k in cand}
                suf = propose_clean([(Q[k], S[k], gen[k][:cut[k]], len(gen[k]) - cut[k]) for k in cand], final=False)
                props = {k: gen[k][:cut[k]] + s for k, s in zip(cand, suf) if s}   # skipped -> keep current
                cand = [k for k in cand if k in props]
                sc = score([(Q[k], props[k]) for k in cand])
                for k, s in zip(cand, sc):
                    att += 1
                    if s > cur[k]:
                        gen[k], cur[k] = props[k], s
                        acc += 1
            for k in act:
                hit = [p for p, t in enumerate(gen[k]) if t in stop_ids]
                if hit:
                    gen[k] = gen[k][:hit[0]]
                    alive[k] = False
            print(f'  batch {b0}: block {blk + 1}/{args.block_num}, alive {sum(alive)}/{len(idxs)}, '
                  f'accept {acc}/{att}, {time.time() - t0:.0f}s', flush=True)
        with open(args.out, 'a') as f:
            for k, i in enumerate(idxs):
                resp = tok.decode(gen[k], skip_special_tokens=False)
                # a response cut at the token budget gets no end-of-turn token: the model never ended it
                end = '<|im_end|>\n' if not alive[k] else ''
                text = f'<|im_start|>user\n{Q[k]}<|im_end|>\n<|im_start|>assistant\n{resp}{end}'
                f.write(json.dumps({'idx': i, 'text': text, 'finished': not alive[k],
                                    'n_tokens': len(gen[k]), 'score': cur[k], 'leak': bool(LEAK.search(resp)),
                                    'solution': S[k]}) + '\n')
        print(f'batch {b0}: {len(idxs)} problems in {time.time() - t0:.0f}s, '
              f'finished {sum(not a for a in alive)}/{len(idxs)}', flush=True)


if __name__ == '__main__':
    main()
