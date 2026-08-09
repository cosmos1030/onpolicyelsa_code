"""
Build unified JSONL for GMP NTP+KD training:
  80% OpenThoughts3-1.2M  (chat CoT traces, prompt masked)
  20% FineWeb-Edu          (plain pretrain text, full gradient)

Output format per row:
  OpenThoughts3:
    {"text": "<chat-templated full conversation>"}
  FineWeb-Edu:
    {"text": "<raw plain text>", "pretrain": true}

Tokenization/chat-templating is parallelized across CPU cores via
datasets.Dataset.map(num_proc=...) — this is pure CPU work (no model forward
pass), so num_proc should track --cpus-per-task, not GPU count.

Usage:
    python scripts/build_ot3_fineweb_dataset.py \
        --nsamples 200000 \
        --out_path data/ot3_fineweb_200k.jsonl \
        --model_path <tokenizer path> \
        --num_proc 16
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

_ROLE_MAP = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}


def _render_ot(batch, tok, min_tokens, seqlen=None, strip_think=False):
    # strip_think: if the fully-rendered conversation exceeds `seqlen` tokens,
    # drop the <think>...</think> block from the assistant's final turn and
    # keep only the text after it (the model's own post-think write-up),
    # instead of letting main.py's tokenizer truncate from the front later.
    # Matches the mechanism reverse-engineered from the pre-08-04
    # ot3_fineweb_20k.jsonl build (verified byte-for-byte against raw
    # OpenThoughts3-1.2M: the old dataset's think-less rows are an exact
    # match for `assistant_value.split("</think>")[-1].strip()`) -- 62%
    # of its rows already fit as-is and kept <think>, the rest were rescued
    # this way. Domain mix is untouched since nothing here filters rows by
    # length, only rewrites content.
    texts = []
    stripped_flags = []
    for convs in batch["conversations"]:
        if isinstance(convs, str):
            convs = json.loads(convs)
        msgs = [{"role": _ROLE_MAP.get(c["from"], c["from"]), "content": c["value"]} for c in convs]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            texts.append(None)
            stripped_flags.append(False)
            continue
        did_strip = False
        if strip_think and seqlen is not None:
            n_tok = len(tok(text, add_special_tokens=False).input_ids)
            if n_tok > seqlen:
                for m in reversed(msgs):
                    if m["role"] == "assistant":
                        think_end = m["content"].rfind("</think>")
                        if think_end != -1:
                            stripped_content = m["content"][think_end + len("</think>"):].strip()
                            if stripped_content:
                                m["content"] = stripped_content
                                did_strip = True
                        break
                if did_strip:
                    try:
                        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    except Exception:
                        did_strip = False
        texts.append(text)
        stripped_flags.append(did_strip)
    n_tokens = [
        len(tok(t, add_special_tokens=False).input_ids) if t is not None else 0
        for t in texts
    ]
    return {"text": texts, "n_tokens": n_tokens, "stripped_think": stripped_flags}


def _measure_fw(batch, tok):
    n_tokens = [len(tok(t, add_special_tokens=False).input_ids) for t in batch["text"]]
    return {"n_tokens": n_tokens}


def _pack_fw(texts, tok, n_fw, target_seqlen):
    # Replicates ReasoningQAT's (github.com/yasu0001/ReasoningQAT) FineWeb-Edu
    # packing exactly: concatenate raw docs (bos + text + eos) into a running
    # buffer until it tokenizes to >= target_seqlen, then cut the first
    # target_seqlen tokens as one packed sample and start a fresh buffer --
    # instead of filtering/keeping individual (mostly too-short) documents.
    bos = tok.bos_token or ""
    eos = tok.eos_token or ""
    fw_records = []
    buffer = ""
    for text in texts:
        buffer += bos + text + eos
        ids = tok(buffer, add_special_tokens=False).input_ids
        if len(ids) >= target_seqlen:
            packed_text = tok.decode(ids[:target_seqlen])
            fw_records.append({"text": packed_text, "pretrain": True})
            buffer = ""
            if len(fw_records) >= n_fw:
                break
    return fw_records


def build(nsamples: int, out_path: str, model_path: str, seed: int = 42,
          min_tokens: int = 64, num_proc: int = 8, seqlen: int = 2048,
          strip_think_if_long: bool = False, pack_fineweb: bool = False,
          ot_ratio: float = 0.8):
    n_ot = int(nsamples * ot_ratio)
    n_fw = nsamples - n_ot

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    records = []

    # ── OpenThoughts3 ─────────────────────────────────────────────────────────
    print(f"Loading OpenThoughts3 (target {n_ot} samples)...", flush=True)
    ds_ot = load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train")
    ds_ot = ds_ot.shuffle(seed=seed).select(range(min(n_ot * 3, len(ds_ot))))
    ds_ot = ds_ot.map(
        lambda batch: _render_ot(batch, tok, min_tokens, seqlen=seqlen, strip_think=strip_think_if_long),
        batched=True, batch_size=256, num_proc=num_proc,
        remove_columns=[c for c in ds_ot.column_names if c != "conversations"],
        desc="Rendering OT3 chat template",
    )
    ds_ot = ds_ot.filter(lambda ex: ex["text"] is not None and ex["n_tokens"] >= min_tokens,
                         desc="Filtering OT3 by length")
    if strip_think_if_long:
        n_stripped = sum(ds_ot["stripped_think"][:n_ot])
        still_over = sum(1 for n in ds_ot["n_tokens"][:n_ot] if n > seqlen)
        print(f"  think-stripped (exceeded {seqlen} tok as-is): {n_stripped}/{min(n_ot, len(ds_ot))}", flush=True)
        print(f"  still over {seqlen} tok after stripping (will be front-truncated at train time): {still_over}", flush=True)
    ot_records = [{"text": t} for t in ds_ot["text"][:n_ot]]
    print(f"  → {len(ot_records)} OT samples", flush=True)
    records.extend(ot_records)

    # ── FineWeb-Edu ───────────────────────────────────────────────────────────
    if n_fw > 0:
        print(f"Loading FineWeb-Edu (target {n_fw} samples)...", flush=True)
        ds_fw = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train")
        ds_fw = ds_fw.shuffle(seed=seed).select(range(min(n_fw * 3, len(ds_fw))))
        ds_fw = ds_fw.map(
            lambda batch: _measure_fw(batch, tok),
            batched=True, batch_size=256, num_proc=num_proc,
            desc="Measuring FineWeb-Edu length",
        )
        if pack_fineweb:
            fw_records = _pack_fw(ds_fw["text"], tok, n_fw, seqlen)
        else:
            ds_fw = ds_fw.filter(lambda ex: ex["n_tokens"] >= min_tokens, desc="Filtering FineWeb by length")
            fw_records = [{"text": t, "pretrain": True} for t in ds_fw["text"][:n_fw]]
        print(f"  → {len(fw_records)} FineWeb samples", flush=True)
        records.extend(fw_records)
    else:
        print("ot_ratio=1.0 -- skipping FineWeb-Edu entirely", flush=True)

    import random
    random.Random(seed).shuffle(records)
    print(f"Total: {len(records)} samples. Writing to {out_path}...", flush=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Done.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsamples", type=int, default=20000)
    parser.add_argument("--out_path", default="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_20k.jsonl")
    parser.add_argument("--model_path",
                        default="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--ot_ratio", type=float, default=0.8,
                        help="Fraction of --nsamples drawn from OpenThoughts3 (rest from FineWeb-Edu). "
                             "Set to 1.0 to replicate ReasoningQAT's Stage 2 (end-to-end distillation) recipe, "
                             "which trains on 100% OpenThoughts3 with no FineWeb-Edu mixed in at all.")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Token budget used to decide whether to strip <think> (only relevant with --strip_think_if_long).")
    parser.add_argument("--strip_think_if_long", action="store_true",
                        help="If a rendered OT3 conversation exceeds --seqlen tokens, drop its <think>...</think> "
                             "block and keep only the text after it (verified match for how the old, better-scoring "
                             "pre-08-04 ot3_fineweb_20k.jsonl was built). Does not filter/exclude any row by length, "
                             "so domain mix (math/code/science) is unaffected -- only content length changes.")
    parser.add_argument("--pack_fineweb", action="store_true",
                        help="Replicate ReasoningQAT's FineWeb-Edu handling exactly: concatenate raw docs into a "
                             "running buffer (bos+text+eos) until it reaches --seqlen tokens, then cut the first "
                             "--seqlen tokens as one packed sample, instead of filtering/keeping individual "
                             "(mostly too-short) documents by --min_tokens.")
    args = parser.parse_args()
    build(args.nsamples, args.out_path, args.model_path, args.seed, args.min_tokens, args.num_proc,
          args.seqlen, args.strip_think_if_long, args.pack_fineweb, args.ot_ratio)


if __name__ == "__main__":
    main()
