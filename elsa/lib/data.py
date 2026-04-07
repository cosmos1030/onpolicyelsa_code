from absl import logging
import os
import json
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer

def _get_raw_dataset(dataset_name, data_type="train", data_path=None):
    """
    Loads the raw text data.
    If `data_path` is provided, it loads specific json.gz files from that local directory.
    Otherwise, it downloads from the Hugging Face Hub.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if "c4" in dataset_name.lower():
        split_name = "train" if data_type == "train" else "validation"
        data_files_config = {
            "train": "en/c4-train.00000-of-01024.json.gz",
            "validation": "en/c4-validation.00000-of-00008.json.gz"
        }
        if data_path:
            if local_rank == 0:
                logging.info(f"Loading C4 raw data from local path: {data_path}")
            files_to_load = {split: os.path.join(data_path, fname) for split, fname in data_files_config.items()}
            return load_dataset(
                'json',
                data_files={split_name: files_to_load[split_name]},
                split=split_name,
            )
        else:
            if local_rank == 0:
                logging.info("Loading C4 raw data from Hugging Face Hub.")
            return load_dataset(
                'allenai/c4',
                data_files={split_name: data_files_config[split_name]},
                split=split_name,
                trust_remote_code=True,
                cache_dir='~/.cache/huggingface/datasets'
            )
            
    elif "wikitext2" in dataset_name.lower():
        split_name = "train" if data_type == "train" else "test"
        return load_dataset('wikitext', 'wikitext-2-raw-v1', split=split_name, trust_remote_code=True)
    elif "ptb" in dataset_name.lower():
        split_name = "train" if data_type == "train" else "validation"
        return load_dataset('ptb_text_only', 'penn_treebank', split=split_name, trust_remote_code=True)
    elif "trace" in dataset_name.lower():
        # math_trace, code_trace, etc. — loaded from local disk via data_path
        # Supports both a saved HF dataset directory and a .jsonl file
        assert data_path is not None, f"--data_path must be provided for dataset '{dataset_name}'"
        if local_rank == 0:
            logging.info(f"Loading '{dataset_name}' from local path: {data_path}")
        if data_path.endswith('.jsonl') or data_path.endswith('.json'):
            return load_dataset('json', data_files=data_path, split='train')
        else:
            return load_from_disk(data_path)
    elif "math_prompt" in dataset_name.lower():
        # Prompt-only JSONL (uses 'prompt' field). No data_path needed if passed via data_path.
        assert data_path is not None, f"--data_path must be provided for dataset '{dataset_name}'"
        if local_rank == 0:
            logging.info(f"Loading math prompts from: {data_path}")
        return load_dataset('json', data_files=data_path, split='train')
    elif "math_cot" in dataset_name.lower():
        # CoT-only JSONL: reads 'text' field (= problem\n\ncot), strips the problem part.
        assert data_path is not None, f"--data_path must be provided for dataset '{dataset_name}'"
        if local_rank == 0:
            logging.info(f"Loading math CoT data from: {data_path}")
        return load_dataset('json', data_files=data_path, split='train')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def _process_and_tokenize(raw_dataset, dataset_name, tokenizer, nsamples, seqlen, seed):
    random.seed(seed)
    
    all_tokens = []
    if dataset_name.lower() == "c4":
        for _ in tqdm(range(nsamples), desc="Tokenizing C4"):
            while True:
                i = random.randint(0, len(raw_dataset) - 1)
                try:
                    text = raw_dataset[i]['text']
                    tokens = tokenizer(text, return_tensors='pt').input_ids
                    if tokens.shape[1] > seqlen:
                        all_tokens.append(tokens)
                        break
                except Exception:
                    continue # Skip samples that cause tokenization errors
    elif "trace" in dataset_name.lower():
        # Support both Arrow format (generations list) and JSONL format (completion string)
        for sample in tqdm(raw_dataset, desc=f"Tokenizing {dataset_name}"):
            if sample.get('generations'):
                gen = sample['generations'][0]
            elif sample.get('completion'):
                gen = sample['completion']
            elif sample.get('text'):
                gen = sample['text']
            else:
                continue
            if not gen:
                continue
            tokens = tokenizer(gen, return_tensors='pt').input_ids
            if tokens.shape[1] > seqlen:
                all_tokens.append(tokens)
        assert len(all_tokens) > 0, f"No valid trace samples found (all shorter than seqlen={seqlen})"
    elif "math_prompt" in dataset_name.lower():
        # Concatenate all prompts into one token stream, then slice seqlen windows
        full_text = "\n\n".join(s['prompt'] for s in raw_dataset if s.get('prompt'))
        all_tokens.append(tokenizer(full_text, return_tensors='pt').input_ids)
    elif "math_cot" in dataset_name.lower():
        # CoT-only: strip the problem (before first \n\n), keep only the generation part
        cot_texts = []
        for s in raw_dataset:
            text = s.get('text', '')
            if not text:
                continue
            # text = "problem\n\ncot_generation" — drop the problem part
            sep = text.find('\n\n')
            cot = text[sep + 2:] if sep != -1 else text
            if cot.strip():
                cot_texts.append(cot)
        full_text = "\n\n".join(cot_texts)
        all_tokens.append(tokenizer(full_text, return_tensors='pt').input_ids)
    else: # wikitext2, ptb
        text_column = "text" if "wikitext" in dataset_name.lower() else "sentence"
        full_text = "\n\n".join(raw_dataset[text_column])
        all_tokens.append(tokenizer(full_text, return_tensors='pt').input_ids)

    processed_samples = []
    is_gemma_tokenizer = 'gemma' in tokenizer.__class__.__name__.lower()
    for _ in tqdm(range(nsamples), desc="Generating samples"):
        token_source = random.choice(all_tokens)
        
        slice_len = seqlen - 1 if is_gemma_tokenizer else seqlen
        
        start_index = random.randint(0, token_source.shape[1] - slice_len - 1)
        end_index = start_index + slice_len
        
        inp = token_source[:, start_index:end_index]

        if is_gemma_tokenizer:
            bos_tensor = torch.tensor([[tokenizer.bos_token_id]])
            inp = torch.cat([bos_tensor, inp], dim=1)
        
        processed_samples.append({
            "input_ids": inp.squeeze(0).tolist(),
            "attention_mask": [1] * seqlen,
            'labels': inp.squeeze(0).tolist()
        })
            
    return Dataset.from_list(processed_samples)

def get_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    nsamples: int,
    seed: int,
    seqlen: int,
    data_type: str = "train",
    cache_dir: str = "dataset",
    save_to_cache: bool = False,
    data_path: str = None
) -> Dataset:
    """
    Creates or loads a tokenized dataset.
    The `data_path` argument now controls where the RAW data is loaded from.
    The processed (tokenized) data is still cached locally for speed.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    safe_model_name = tokenizer.name_or_path.replace("/", "_")
    path_suffix = f"local_{Path(data_path).name}" if data_path else "hub"
    data_dir = Path(cache_dir) / safe_model_name / data_type / f"{dataset_name}_{nsamples}_{seqlen}_{path_suffix}"
    
    if data_dir.exists():
        if local_rank == 0:
            logging.info(f"Loading cached processed dataset from {data_dir}")
        return load_from_disk(str(data_dir))

    if local_rank == 0:
        logging.info(f"No valid processed cache found. Generating dataset...")
    raw_dataset = _get_raw_dataset(dataset_name, data_type, data_path=data_path)
    dataset = _process_and_tokenize(raw_dataset, dataset_name, tokenizer, nsamples, seqlen, seed)

    if save_to_cache:
        data_dir.mkdir(parents=True, exist_ok=True)
        if local_rank == 0:
            logging.info(f"Saving processed dataset to {data_dir}")
        dataset.save_to_disk(str(data_dir))
        
        metadata_file = data_dir / "metadata.json"
        metadata = {
            'tokenizer_name': tokenizer.name_or_path,
            'dataset_name': dataset_name,
            'data_type': data_type,
            'nsamples': nsamples,
            'seqlen': seqlen,
            'created_at': datetime.now().isoformat(),
            'source_data_path': data_path if data_path else 'Hugging Face Hub'
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    return dataset


class TokenizerWrapper:
    """Wrapper for tokenized input IDs to maintain backward compatibility."""
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids

def get_loaders(
    name: str,
    nsamples: int = 128,
    seed: int = 0,
    seqlen: int = 2048,
    tokenizer: AutoTokenizer = None,
    data_path: str = None
) -> tuple[list, TokenizerWrapper]:
    """
    Provides data in the legacy format for local pruning solvers.
    This function is now a wrapper around the new `get_dataset` function.

    Returns:
        tuple: (trainloader, testenc)
            - trainloader: A list of (input_tensor, target_tensor) tuples.
            - testenc: A TokenizerWrapper object containing test data.
    """
    data_path = None
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        logging.info("Using `get_loaders` (legacy compatibility mode).")

    train_dataset = get_dataset(
        dataset_name=name,
        tokenizer=tokenizer,
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        data_type="train",
        data_path=data_path
    )

    trainloader = []
    for sample in train_dataset:
        inp = torch.tensor(sample['input_ids']).unsqueeze(0)  # (seqlen,) -> (1, seqlen)
        if 'gemma' in tokenizer.__class__.__name__.lower():
            inp[:, 0] = tokenizer.bos_token_id
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    if name.lower() == "c4":
        valdata = _get_raw_dataset("c4", "validation",data_path=data_path)
        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors='pt')
        valenc = valenc.input_ids[:,:(256*seqlen)]
        valenc = TokenizerWrapper(valenc)
    elif name.lower() == "wikitext2":
        valdata = _get_raw_dataset("wikitext2", "validation",data_path=data_path)
        valenc = tokenizer("\n\n".join(valdata["text"]), return_tensors='pt')
        valenc = TokenizerWrapper(valenc.input_ids)

    return trainloader, valenc

