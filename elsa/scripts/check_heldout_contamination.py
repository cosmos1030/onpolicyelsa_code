"""Are the 'held-out' evaluation prompts actually absent from the training file?

policy_divergence_tsne.py enforces held-out by INDEX: the calibration pool is
indices 0..95,999 of OpenThoughts3 shuffled with seed 42, evaluation starts at
200,000, and an assert guards the gap. Index-disjoint is not content-disjoint --
OpenThoughts3 aggregates many sources, so the same problem can sit at several
indices, and a prompt drawn from 200,000 can be a duplicate of one the model
trained on.

Matching has to happen on decoded text. The training file is JSONL whose rows
are {"text": "<|im_start|>user ..."}, so backslashes are escaped on the raw
line: a needle containing LaTeX will never match the line as bytes. A first
pass that scanned raw lines reported 4 of 30 contaminated, and a second pass
with longer needles taken from mid-text reported 0 of 30 -- the disagreement was
entirely this, because the four that matched were the ones whose opening
sentence happens to contain no backslash. This decodes each row and compares
normalised text.

Usage: check_heldout_contamination.py [--train FILE] [--states_dir DIR]
"""
import argparse
import json
import re


def norm(s):
    # Collapse whitespace and drop chat-template control tokens, so the same
    # problem written with different line wrapping still matches.
    s = re.sub(r"<\|[^|]*\|>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def body(prompt):
    return norm(prompt.split("<|im_start|>user", 1)[-1])


def needles(text, n, k):
    """k evenly spaced windows of n chars, so one unlucky window cannot decide
    the verdict and a partial overlap still shows up."""
    t = text
    if len(t) <= n:
        return [t] if t else []
    out, step = [], max(1, (len(t) - n) // max(1, k - 1))
    for i in range(k):
        a = min(i * step, len(t) - n)
        out.append(t[a:a + n])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",
                    default="/home1/doyoonkim/projects/elsa/data/"
                            "ot3_fineweb_40k_qwen3_nostrip_8192.jsonl")
    ap.add_argument("--states_dir",
                    default="/home1/doyoonkim/projects/elsa/logs/"
                            "policy_divergence/n30_k64_core")
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    pc = json.load(open(f"{a.states_dir}/prompts.json"))
    ids = pc["ids"]
    P = [needles(body(p), a.n, a.k) for p in pc["prompts"]]
    S = [needles(norm(s), a.n, a.k) for s in pc["solutions"]]

    hp = [0] * len(P)
    hs = [0] * len(S)
    rows = 0
    for line in open(a.train, errors="ignore"):
        try:
            t = json.loads(line)["text"]
        except Exception:
            continue
        rows += 1
        L = norm(t)
        for i, nds in enumerate(P):
            if any(nd in L for nd in nds):
                hp[i] += 1
        for i, nds in enumerate(S):
            if any(nd in L for nd in nds):
                hs[i] += 1

    print(f"scanned {rows} decoded training rows; "
          f"{a.k} needles x {a.n} chars per item\n")
    print(f"{'#':>3} {'id':<22}{'prompt':>8}{'CoT':>6}")
    bad = []
    for i, pid in enumerate(ids):
        flag = ""
        if hp[i] or hs[i]:
            bad.append(i)
            flag = "  <-- CONTAMINATED"
        print(f"{i:>3} {pid:<22}{hp[i]:>8}{hs[i]:>6}{flag}")

    n = len(ids)
    clean = [i for i in range(n) if i not in bad]
    print(f"\ncontaminated: {len(bad)}/{n} -> {bad}")
    print(f"clean       : {len(clean)}/{n} -> {clean}")
    print(f"first 6 (fixed-CoT panel) contaminated: "
          f"{[i for i in bad if i < 6]}")

    if a.out:
        json.dump({"ids": ids, "prompt_hits": hp, "cot_hits": hs,
                   "contaminated": bad, "clean": clean},
                  open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
