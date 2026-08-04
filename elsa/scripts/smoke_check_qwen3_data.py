"""
Lightweight smoke check for ot3_fineweb_*_qwen3.jsonl: reads only the first N
lines directly (itertools.islice), does NOT instantiate MixedTextDataset /
MixedPromptDataset (those read the whole file into memory regardless of the
nsamples argument, which OOMs on quick smoke tests).
"""
import argparse
import itertools
import json

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl")
    parser.add_argument("--model_path",
                        default="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    n_ot, n_fw = 0, 0
    with open(args.data_path) as f:
        for i, line in enumerate(itertools.islice(f, args.n)):
            rec = json.loads(line)
            text = rec.get("text", "")
            is_pretrain = rec.get("pretrain", False)
            n_fw += is_pretrain
            n_ot += not is_pretrain
            if i < 3:
                print(f"--- record {i} (pretrain={is_pretrain}) ---")
                print("raw text[:200]:", repr(text[:200]))
                ids = tok(text, truncation=True, max_length=64, add_special_tokens=False).input_ids
                print("decoded first 64 tokens:", repr(tok.decode(ids)))
                if not is_pretrain and "<think>" in text:
                    prompt = text.split("<think>")[0].strip()
                    print("MixedPromptDataset-style prompt fallback[:150]:", repr(prompt[:150]))
                print()

    print(f"Checked {n_ot + n_fw} records: {n_ot} OT (chat-rendered), {n_fw} FW (pretrain)")


if __name__ == "__main__":
    main()
