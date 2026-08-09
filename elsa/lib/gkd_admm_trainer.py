"""
GKDADMMTrainer: ADMM pruning with on-policy knowledge distillation loss.
Inherits ADMMTrainer to keep ADMM mechanics intact, replaces NTP loss with KD.
"""
import collections
import hashlib
import os
import pickle
import random
import time

import wandb
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import GenerationConfig
from .trainer import ADMMTrainer, compute_self_distillation_loss
from absl import logging
import json


_DEFAULT_DATASET_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "datasets")


def _tokenizer_identity(tokenizer):
    """Content-based tokenizer fingerprint instead of the checkpoint path —
    Qwen3-1.7B/4B/8B ship byte-identical tokenizers under different snapshot
    directories, so keying the dataset cache on `tokenizer.name_or_path`
    forced a redundant multi-hour re-tokenization per model size even though
    the cached samples would have been identical. Falls back to the path for
    slow (non-Rust-backed) tokenizers that don't expose backend_tokenizer.
    """
    try:
        return hashlib.md5(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    except AttributeError:
        return tokenizer.name_or_path


def _dataset_cache_path(cache_dir, jsonl_path, tokenizer_name, **kwargs):
    key = f"{jsonl_path}|{tokenizer_name}|" + "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{h}.pkl")


# ---------------------------------------------------------------------------
# Dataset: prompt + gold answer from local JSONL (with pre-extracted answers)
# ---------------------------------------------------------------------------
class MathPromptWithAnswerDataset(Dataset):
    """
    Loads math prompts and gold answers from a local JSONL file.

    Expected JSONL format (one JSON object per line):
      {"prompt": "<full prompt with chat template + <think>\\n>", "answer": "<gold answer>"}
    OR (fallback, constructs prompt via apply_chat_template):
      {"problem": "<problem text>", "answer": "<gold answer>"}

    Use data/math_220k_with_answers.jsonl (pre-built from math_220k_prompts + cot).
    """
    THINK_TAG = "<think>\n"

    def __init__(self, jsonl_path, tokenizer, max_prompt_len=512,
                 nsamples=None, seed=42, cache_dir=_DEFAULT_DATASET_CACHE_DIR,
                 hf_dataset_name=None):  # hf_dataset_name kept for API compat, ignored

        cache_path = _dataset_cache_path(
            cache_dir, jsonl_path, _tokenizer_identity(tokenizer),
            cls="MathPromptWithAnswerLocal", max_prompt_len=max_prompt_len,
            nsamples=nsamples, seed=seed,
        )
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

        if is_rank0:
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        self.samples = pickle.load(f)
                    logging.info(
                        f"MathPromptWithAnswerDataset: loaded {len(self.samples)} samples from cache"
                    )
                    if dist.is_initialized():
                        dist.barrier()
                    return
                except (EOFError, pickle.UnpicklingError):
                    logging.warning("MathPromptWithAnswerDataset: corrupted cache, rebuilding")
                    os.remove(cache_path)

        random.seed(seed)
        if is_rank0:
            with open(jsonl_path) as f:
                records = [json.loads(l) for l in f]
        else:
            records = []

        if nsamples and nsamples < len(records):
            records = random.sample(records, nsamples)

        self.samples = []
        skipped = 0
        for rec in records:
            gold = rec.get("answer", "")
            if not gold:
                skipped += 1
                continue

            # 로컬 파일에 이미 prompt 있으면 그대로 사용, 없으면 chat template 적용
            if "prompt" in rec:
                prompt_text = rec["prompt"]
            elif "problem" in rec:
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": rec["problem"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                ) + self.THINK_TAG
            else:
                skipped += 1
                continue

            enc = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_prompt_len,
                return_tensors="pt",
                padding=False,
            )
            self.samples.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "gold_answer":    gold.strip(),
            })

        logging.info(
            f"MathPromptWithAnswerDataset: {len(self.samples)} samples "
            f"({skipped} skipped)"
        )
        if is_rank0:
            with open(cache_path, "wb") as f:
                pickle.dump(self.samples, f)

        if dist.is_initialized():
            dist.barrier()

        if not is_rank0:
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)

    @staticmethod
    def _extract_boxed(text: str):
        """Brace-balanced extraction of last \\boxed{} content."""
        idx = text.rfind(r'\boxed{')
        if idx == -1:
            return None
        depth = 0
        start = idx + len(r'\boxed{')
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                if depth == 0:
                    return text[start:i].strip()
                depth -= 1
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_prompts_with_answers(pad_token_id):
    """collate_fn for MathPromptWithAnswerDataset — left-pads, keeps gold strings."""
    def _collate(batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)
        input_ids_list, mask_list, golds = [], [], []
        for x in batch:
            pad_len = max_len - x["input_ids"].shape[0]
            input_ids_list.append(
                torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long),
                           x["input_ids"]])
            )
            mask_list.append(
                torch.cat([torch.zeros(pad_len, dtype=torch.long),
                           x["attention_mask"]])
            )
            golds.append(x["gold_answer"])
        return {
            "input_ids":      torch.stack(input_ids_list),
            "attention_mask": torch.stack(mask_list),
            "gold_answers":   golds,          # list[str], length B
        }
    return _collate


# ---------------------------------------------------------------------------
# Dataset: prompt-only from math 220k JSONL
# ---------------------------------------------------------------------------
class MixedPromptDataset(Dataset):
    """
    Loads math prompts from a JSONL file (uses 'prompt' field, chat-template applied).
    Returns tokenized prompt tensors for on-policy generation.
    """
    def __init__(self, jsonl_path, tokenizer, max_prompt_len=512, nsamples=None, seed=42,
                 cache_dir=_DEFAULT_DATASET_CACHE_DIR):
        cache_path = _dataset_cache_path(cache_dir, jsonl_path, _tokenizer_identity(tokenizer),
                                         cls="MathPrompt", max_prompt_len=max_prompt_len,
                                         nsamples=nsamples, seed=seed)
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

        if is_rank0:
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        self.samples = pickle.load(f)
                    if len(self.samples) == 0:
                        logging.warning(f"MixedPromptDataset: empty cache {cache_path}, rebuilding")
                        os.remove(cache_path)
                    else:
                        logging.info(f"MixedPromptDataset: loaded {len(self.samples)} samples from cache {cache_path}")
                        if dist.is_initialized():
                            dist.barrier()
                        return
                except (EOFError, pickle.UnpicklingError):
                    logging.warning(f"MixedPromptDataset: corrupted cache {cache_path}, rebuilding")
                    os.remove(cache_path)

        random.seed(seed)
        if is_rank0:
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f if line.strip()]
        else:
            records = []

        random.shuffle(records)
        if nsamples and nsamples < len(records):
            records = records[:nsamples]

        self.samples = []
        for rec in records:
            prompt = rec.get("prompt", "")
            if not prompt:
                # fallback: extract prompt from 'text' field by splitting on <think>
                text = rec.get("text", "")
                prompt = text.split("<think>")[0].strip() if "<think>" in text else ""
            if not prompt:
                continue
            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_prompt_len,
                return_tensors="pt",
                padding=False,
            )
            self.samples.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

        if is_rank0:
            with open(cache_path, "wb") as f:
                pickle.dump(self.samples, f)
            logging.info(f"MixedPromptDataset: {len(self.samples)} prompts loaded and cached to {cache_path}")

        if dist.is_initialized():
            dist.barrier()

        if not is_rank0:
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)
            logging.info(f"MixedPromptDataset: loaded {len(self.samples)} samples from cache {cache_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_prompts(pad_token_id):
    """Returns a collate_fn that left-pads prompts to equal length."""
    def _collate(batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)
        input_ids_list, mask_list = [], []
        for x in batch:
            pad_len = max_len - x["input_ids"].shape[0]
            input_ids_list.append(
                torch.cat([torch.full((pad_len,), pad_token_id, dtype=torch.long),
                           x["input_ids"]])
            )
            mask_list.append(
                torch.cat([torch.zeros(pad_len, dtype=torch.long),
                           x["attention_mask"]])
            )
        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(mask_list),
            "prompt_len": torch.tensor(max_len),
        }
    return _collate


# ---------------------------------------------------------------------------
# Dataset: CoT text for NTP + prompt for KD generation
# ---------------------------------------------------------------------------
class MixedTextDataset(Dataset):
    """
    Loads math CoT traces from math_220k_cot.jsonl for hybrid NTP + KD training.

    Each sample provides:
      - input_ids / attention_mask / labels: full CoT text for NTP loss
        (problem portion is masked with -100 in labels)
      - prompt_ids / prompt_mask: problem + '<think>' prefix for on-policy generation

    text format: "problem\n\n<think>CoT</think>answer"
    Split point: '<think>' tag — prompt = text[:idx+len('<think>')], cot = text[idx:]
    """
    THINK_TAG = "<think>"
    MIN_SAMPLE_LEN = 128  # drop shorter samples (see skip check in __init__)

    def __init__(self, jsonl_path, tokenizer, max_len=2048, max_prompt_len=512,
                 nsamples=None, seed=42, cache_dir=_DEFAULT_DATASET_CACHE_DIR,
                 append_eos=False):
        cache_path = _dataset_cache_path(cache_dir, jsonl_path, _tokenizer_identity(tokenizer),
                                         cls="MathCotKD" if not append_eos else "MathCotKD_eos",
                                         max_len=max_len,
                                         max_prompt_len=max_prompt_len,
                                         nsamples=nsamples, seed=seed,
                                         min_len=self.MIN_SAMPLE_LEN)
        is_distributed = dist.is_initialized()
        is_rank0 = (not is_distributed) or dist.get_rank() == 0

        # Only rank 0 touches the filesystem (check/build/write cache); other ranks
        # get the result via an NCCL broadcast instead of re-reading the same path
        # off disk. Relying on all ranks to independently read cache_path right
        # after a barrier is a real race on network filesystems — rank 0's write
        # is not guaranteed to be visible to other processes the instant its
        # barrier call returns, which can (and did) throw FileNotFoundError.
        self.samples = None
        if is_rank0:
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        self.samples = pickle.load(f)
                    logging.info(f"MixedTextDataset: loaded {len(self.samples)} samples from cache {cache_path}")
                except (EOFError, pickle.UnpicklingError):
                    logging.warning(f"MixedTextDataset: corrupted cache {cache_path}, rebuilding")
                    os.remove(cache_path)

        if is_rank0 and self.samples is None:
            random.seed(seed)
            with open(jsonl_path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            random.shuffle(records)
            if nsamples and nsamples < len(records):
                records = records[:nsamples]

            self.samples = []
            from tqdm import tqdm
            for rec in tqdm(records, desc=f"MixedTextDataset: tokenizing {os.path.basename(jsonl_path)}"):
                text = rec.get("text", "")
                if not text:
                    continue

                is_pretrain = rec.get("pretrain", False)

                if is_pretrain:
                    # Pretrain data (e.g. FineWeb-Edu): no prompt masking, gradient flows everywhere.
                    prompt_text = ""
                    prompt_len = 0
                else:
                    # Split at <think>
                    idx = text.find(self.THINK_TAG)
                    if idx == -1:
                        # fallback: split at first double-newline
                        idx = text.find("\n\n")
                        if idx == -1:
                            continue
                        prompt_text = text[:idx]
                    else:
                        prompt_text = text[:idx + len(self.THINK_TAG)]

                    cot_text = text[len(prompt_text):]
                    if not cot_text.strip():
                        continue

                # Full sequence for NTP
                full_enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_len - 1 if append_eos else max_len,
                    return_tensors="pt",
                    padding=False,
                )
                if append_eos:
                    full_ids = torch.cat([
                        full_enc["input_ids"].squeeze(0),
                        torch.tensor([tokenizer.eos_token_id], dtype=torch.long),
                    ])
                    full_mask = torch.cat([
                        full_enc["attention_mask"].squeeze(0),
                        torch.tensor([1], dtype=torch.long),
                    ])
                else:
                    full_ids = full_enc["input_ids"].squeeze(0)
                    full_mask = full_enc["attention_mask"].squeeze(0)

                # Skip very short samples: a long run of ~max_len batches
                # followed by an abrupt drop to a short one (e.g. an
                # 82-token FineWeb-Edu snippet) reproducibly triggered a
                # CUBLAS_STATUS_INTERNAL_ERROR on every one of 3 independent
                # runs at the identical training step — a cuBLAS shape-
                # transition issue, not corrupted data (confirmed by
                # decoding the exact crashing batch). No sequence packing
                # implemented here, so just drop the outliers instead.
                if full_ids.shape[0] < self.MIN_SAMPLE_LEN:
                    continue

                # Prompt for KD generation
                if is_pretrain:
                    prompt_ids = torch.zeros(0, dtype=torch.long)
                    prompt_mask_t = torch.zeros(0, dtype=torch.long)
                else:
                    prompt_enc = tokenizer(
                        prompt_text,
                        truncation=True,
                        max_length=max_prompt_len,
                        return_tensors="pt",
                        padding=False,
                    )
                    prompt_ids = prompt_enc["input_ids"].squeeze(0)
                    prompt_mask_t = prompt_enc["attention_mask"].squeeze(0)
                    prompt_len = prompt_ids.shape[0]

                # Labels: mask problem tokens with -100 (pretrain: no masking)
                labels = full_ids.clone()
                if not is_pretrain:
                    labels[:prompt_len] = -100

                self.samples.append({
                    "input_ids": full_ids,
                    "attention_mask": full_mask,
                    "labels": labels,
                    "prompt_ids": prompt_ids,
                    "prompt_mask": prompt_mask_t,
                })

            # Atomic write (temp + rename) so a concurrent reader never sees a
            # partially-written or momentarily-missing file.
            tmp_path = f"{cache_path}.tmp{os.getpid()}"
            with open(tmp_path, "wb") as f:
                pickle.dump(self.samples, f)
            os.replace(tmp_path, cache_path)
            logging.info(f"MixedTextDataset: {len(self.samples)} samples built and cached to {cache_path}")

        # Other ranks load the cache from disk themselves rather than receiving
        # it via dist.broadcast_object_list — broadcasting a ~200k-sample /
        # tens-of-GB Python object through gloo/NCCL segfaults the whole job
        # (observed repeatedly: crash immediately after rank 0 finishes the
        # build, right at this broadcast call). No dist.barrier() here: a cold
        # rank-0 tokenization build can take multiple hours, which exceeds
        # NCCL's default ~2-hour collective-op watchdog timeout and kills
        # every other rank while they wait at the barrier (observed on 8B).
        # The poll-for-file loop below already provides safe cross-rank
        # synchronization without a blocking collective.
        if is_distributed:
            if not is_rank0:
                # A cold rank-0 build of the full 200k-sample set takes hours
                # (confirmed: ~2.5h for this dataset) — 60 attempts * 2s (2
                # min total) was nowhere near enough and made every other
                # rank give up and crash the instant a cache-key change (e.g.
                # MIN_SAMPLE_LEN) forced a rebuild. Poll for up to 4 hours.
                _max_wait_s = 4 * 3600
                _poll_interval_s = 10
                for _attempt in range(_max_wait_s // _poll_interval_s):
                    if os.path.exists(cache_path):
                        try:
                            with open(cache_path, "rb") as f:
                                self.samples = pickle.load(f)
                            break
                        except (EOFError, pickle.UnpicklingError):
                            pass
                    time.sleep(_poll_interval_s)
                if self.samples is None:
                    raise RuntimeError(f"MixedTextDataset: cache {cache_path} not visible after rank 0 build "
                                        f"(waited {_max_wait_s}s)")
                logging.info(f"MixedTextDataset: loaded {len(self.samples)} samples from cache {cache_path} (non-rank0)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class NTPPromptWrapper(Dataset):
    """
    Thin wrapper over MixedTextDataset that exposes prompt_ids/prompt_mask
    as input_ids/attention_mask so it can be used with collate_prompts /
    generate_chosen_cache while preserving the same sample order as the NTP
    DataLoader (shuffle=False on both sides).
    """
    def __init__(self, ntp_dataset):
        self.dataset = ntp_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        s = self.dataset[idx]
        return {"input_ids": s["prompt_ids"], "attention_mask": s["prompt_mask"]}


def collate_cot_kd(pad_token_id):
    """
    Collate for MixedTextDataset.
    - Right-pads full sequences (input_ids/attention_mask/labels)
    - Left-pads prompts (prompt_ids/prompt_mask) for generation
    """
    def _collate(batch):
        # Full sequence: right-pad
        max_full = max(x["input_ids"].shape[0] for x in batch)
        max_prompt = max(x["prompt_ids"].shape[0] for x in batch)

        input_ids_list, mask_list, labels_list = [], [], []
        prompt_ids_list, prompt_mask_list = [], []

        for x in batch:
            # Right-pad full sequence
            pad_len = max_full - x["input_ids"].shape[0]
            input_ids_list.append(
                torch.cat([x["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
            )
            mask_list.append(
                torch.cat([x["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            labels_list.append(
                torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )
            # Left-pad prompt
            pad_len_p = max_prompt - x["prompt_ids"].shape[0]
            prompt_ids_list.append(
                torch.cat([torch.full((pad_len_p,), pad_token_id, dtype=torch.long),
                           x["prompt_ids"]])
            )
            prompt_mask_list.append(
                torch.cat([torch.zeros(pad_len_p, dtype=torch.long),
                           x["prompt_mask"]])
            )

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(mask_list),
            "labels": torch.stack(labels_list),
            "prompt_ids": torch.stack(prompt_ids_list),
            "prompt_mask": torch.stack(prompt_mask_list),
            "prompt_len": torch.tensor(max_prompt),
        }
    return _collate


# ---------------------------------------------------------------------------
# GKDADMMTrainer
# ---------------------------------------------------------------------------
class GKDADMMTrainer(ADMMTrainer):
    """
    ADMM trainer with on-policy KD loss instead of NTP.

    Per training step:
      1. Generate completion from student (no_grad)
      2. Forward pass: student + teacher on (prompt + completion)
      3. Reverse KL(student || teacher) on generated tokens only
      4. ADMM proximal + Adam step + dual update (inherited from ADMMTrainer)
    """

    def __init__(
        self,
        teacher_model,
        max_new_tokens: int = 512,
        gen_temperature: float = 1.0,
        kd_temperature: float = 1.0,
        ntp_lambda: float = 0.0,
        kd_topk: int = 50,
        kd_interval: int = 1,
        kd_lambda: float = 1.0,
        use_vllm: bool = False,
        vllm_model_name: str = None,
        vllm_gpu_memory_utilization: float = 0.3,
        vllm_max_model_len: int = None,
        kd_buffer_size: int = 0,
        kd_buffer_refresh_interval: int = 32,
        kd_step_interval: int = 1,
        offpolicy_kd: bool = False,
        kd_offpolicy_ntp: bool = False,
        kd_triple_loss: bool = False,
        kd_opkd_lambda: float = 0.0,
        admm_tr_use_opkd_rollout: bool = False,
        generate_with_teacher: bool = False,
        forward_kl: bool = False,
        prompt_dataset=None,
        opd_enabled: bool = False,
        opd_lambda: float = 0.0,
        opd_vllm_max_tokens: int = 256,
        opd_vllm_engine=None,
        opd_vllm_params=None,
        opd_prompt_dataset=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.max_new_tokens = max_new_tokens
        self.gen_temperature = gen_temperature
        self.kd_temperature = kd_temperature
        self.ntp_lambda = ntp_lambda
        self.kd_topk = kd_topk
        self.kd_interval = kd_interval  # run KD every N steps; 1 = every step (ignored when buffer active)
        self.kd_lambda = kd_lambda      # weight for KD loss when combined with NTP
        self.kd_buffer_size = kd_buffer_size                        # 0 = disabled
        # default refresh = buffer drains naturally (buffer_size * step_interval steps)
        if kd_buffer_refresh_interval == 32 and kd_buffer_size > 0 and kd_step_interval > 1:
            kd_buffer_refresh_interval = kd_buffer_size * kd_step_interval
        self.kd_buffer_refresh_interval = kd_buffer_refresh_interval
        self.kd_step_interval = kd_step_interval                    # apply KD every N steps (1 = every step)
        self.offpolicy_kd = offpolicy_kd
        self.kd_offpolicy_ntp = kd_offpolicy_ntp      # NTP + dataset-based KD (no generation)
        self.kd_triple_loss = kd_triple_loss          # NTP + dataset KD + OPKD
        self.kd_opkd_lambda = kd_opkd_lambda          # weight for on-policy KD in triple mode
        self.admm_tr_use_opkd_rollout = admm_tr_use_opkd_rollout
        self._opkd_inputs = None
        self.forward_kl = forward_kl
        self.generate_with_teacher = generate_with_teacher
        self.prompt_dataset = prompt_dataset  # separate prompt source for vLLM buffer (optional)
        self._rollout_buffer: collections.deque = collections.deque()
        self._prompt_pool = None  # lazily built from prompt_dataset or train_dataset
        self._last_kd_step = -1   # tracks last optimizer step that popped from buffer

        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=gen_temperature,
            use_cache=True,  # generation is under no_grad, KV cache is safe
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Move teacher to same device as student; keep frozen
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

        # Per-step state for conditional KD
        self._kd_inputs = None  # set in training_step, read in compute_loss
        # TR global z-projection: flag to run setup_global_tr_z once after optimizer is ready
        self._tr_z_initialized = False

        # OPD (On-Policy Distillation, rollouts generated by the live x model) state
        self.opd_enabled = opd_enabled
        self.opd_lambda = opd_lambda
        self.opd_vllm_max_tokens = opd_vllm_max_tokens
        self._opd_vllm_engine = opd_vllm_engine    # pre-built engine (FSDP+subprocess) or None
        self._opd_vllm_params = opd_vllm_params
        self._opd_prompt_dataset = opd_prompt_dataset
        self._opd_pool: list = []
        self._opd_pool_ptr: int = 0
        self._opd_last_refill_step: int = -1
        self._opd_prompt_iter = None
        self._opd_inputs = None                     # rollout popped per training step

        # vLLM rollout engine (optional)
        self.use_vllm = use_vllm
        self.vllm_engine = None
        self.vllm_sampling_params = None
        if use_vllm:
            if vllm_model_name is None:
                raise ValueError("vllm_model_name must be set when use_vllm=True")
            # max_model_len = prompt + generation (no need for model's full context)
            _vllm_max_model_len = vllm_max_model_len or (max_new_tokens + 1024)
            self._init_vllm_engine(vllm_model_name, vllm_gpu_memory_utilization, _vllm_max_model_len)

    def _get_opd_prompt_iter(self):
        """Lazily build infinite prompt iterator for OPD pool generation."""
        if self._opd_prompt_iter is not None:
            return self._opd_prompt_iter
        ds = self._opd_prompt_dataset
        if ds is None:
            return None
        from torch.utils.data import DataLoader
        _loader = DataLoader(ds, batch_size=1, shuffle=True,
                             collate_fn=collate_prompts(self.tokenizer.pad_token_id or 0))
        def _inf():
            while True:
                for x in _loader:
                    yield x
        self._opd_prompt_iter = _inf()
        return self._opd_prompt_iter

    @torch.no_grad()
    def _generate_opd_pool(self, model):
        """Generate OPD pool using the live (x) model via vLLM.

        Syncs current x weights to vLLM, generates admm_interval×grad_accum rollouts,
        broadcasts to all ranks. Updates self._opd_pool and resets self._opd_pool_ptr.
        Also sets self._tr_current_batch to first 8 rollouts for TR-z KL check.

        Previously this hard-masked onto z (zeroing newly-pruned positions) before
        generating, on the theory that OPD should distill into the model that will
        actually be deployed. But the OPD loss forward pass (_compute_opd_backward_kl)
        always runs on x -- the z-mask here was undone in `finally` before that call --
        so rollouts were sampled from z's distribution while the gradient was computed
        against x's distribution. On-policy distillation requires the sampling policy
        and the policy being updated to match; under fast-growing cosine/cubic z
        schedules where z and x diverge significantly interval-to-interval, this
        mismatch could produce degenerate KL values from tokens that are likely under
        z but near-zero probability under x. Generating from x directly makes this
        genuinely on-policy; the ADMM proximal term is still what pulls x toward the
        sparse target z, so the "distill into the deployed model" goal is served by
        that mechanism instead. This also makes the TR-z KL calibration batch (see
        _compute_kl_with_z) more representative -- it should reflect the CURRENT
        model's own behavior, not a hypothetical already-masked model's.
        """
        from lib.gmp_trainer import _opkd_broadcast_pool, _opkd_pool_to_batch
        from contextlib import nullcontext

        admm_interval = self.args.admm_interval
        grad_accum = self.args.gradient_accumulation_steps
        pool_size = admm_interval * grad_accum

        is_distributed = dist.is_initialized()
        is_main = (not is_distributed) or dist.get_rank() == 0
        device = next(model.parameters()).device

        prompt_iter = self._get_opd_prompt_iter()
        if prompt_iter is None:
            logging.warning("OPD: no prompt dataset configured, skipping pool generation")
            return

        # --- Sync live (x) weights to vLLM and generate rollouts ---
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
            _FSDP_AVAILABLE = True
        except ImportError:
            _FSDP_AVAILABLE = False

        _in_fsdp = _FSDP_AVAILABLE and isinstance(model, FSDP)
        _fsdp_ctx = (FSDP.summon_full_params(model, writeback=False, offload_to_cpu=True, rank0_only=True)
                     if _in_fsdp else nullcontext())

        with _fsdp_ctx:
            if is_main and self._opd_vllm_engine is not None:
                if _in_fsdp and hasattr(self._opd_vllm_engine, 'sync_weights'):
                    _sd = {n: p.data.cpu() for n, p in model.named_parameters()}
                    self._opd_vllm_engine.sync_weights(_sd)
                    del _sd
                elif not _in_fsdp:
                    # Non-FSDP: sync directly via vLLM internal API
                    try:
                        _eng = self._opd_vllm_engine.llm_engine
                        _exec = (_eng.engine_core.model_executor if hasattr(_eng, 'engine_core')
                                 else _eng.model_executor)
                        _vm = _exec.driver_worker.model_runner.model
                        _vs = {k: v for k, v in _vm.named_parameters()}
                        for _n, _p in model.named_parameters():
                            if _n in _vs:
                                _vs[_n].data.copy_(_p.data.to(_vs[_n].dtype))
                    except Exception as _e:
                        logging.warning(f"OPD vLLM weight sync failed: {_e}")

        # Generate rollouts (rank 0 only)
        pool = []
        if is_main and self._opd_vllm_engine is not None:
            from vllm.inputs import TokensPrompt as _TokensPrompt
            pool_batches = [next(prompt_iter) for _ in range(pool_size)]
            vllm_inputs = [
                _TokensPrompt(prompt_token_ids=b['input_ids'][0][:int(b['prompt_len'].item())].tolist())
                for b in pool_batches
            ]
            vllm_outs = self._opd_vllm_engine.generate(vllm_inputs, self._opd_vllm_params)
            for pb, vo in zip(pool_batches, vllm_outs):
                plen = int(pb['prompt_len'].item())
                p_ids = pb['input_ids'][:, :plen].cpu()
                gen_ids = torch.tensor([vo.outputs[0].token_ids], dtype=torch.long)
                full_seq = torch.cat([p_ids, gen_ids], dim=1)
                pool.append({"full_seq": full_seq, "prompt_len": plen})
            logging.info(f"  OPD pool generated: {len(pool)} rollouts (x-generated, z_sp={self._tr_z_sp:.3f})")

        pool = _opkd_broadcast_pool(pool, is_distributed, device)
        self._opd_pool = pool
        self._opd_pool_ptr = 0

        # Use first 8 pool rollouts as TR-z calibration batch
        if pool and getattr(self.args, 'admm_tr_z_proj', False):
            _cal = _opkd_pool_to_batch(pool[:min(8, len(pool))], str(device))
            self._tr_current_batch = _cal

    def _pop_opd_inputs(self, device):
        """Pop one rollout from OPD pool, return as kd_inputs dict or None if empty."""
        if not self._opd_pool or self._opd_pool_ptr >= len(self._opd_pool):
            return None
        item = self._opd_pool[self._opd_pool_ptr]
        self._opd_pool_ptr += 1
        full_seq = item['full_seq'].to(device)
        plen = item['prompt_len']
        attn = (full_seq != self.tokenizer.pad_token_id).long()
        return {
            "input_ids": full_seq,
            "attention_mask": attn,
            "prompt_len": torch.tensor(plen, device=device),
        }

    def _init_vllm_engine(self, model_name: str, gpu_memory_utilization: float, max_model_len: int):
        """Initialize vLLM LLM engine for fast student rollout generation."""
        import os
        os.environ["VLLM_USE_V1"] = "0"  # use V0 engine for direct weight access
        from vllm import LLM, SamplingParams
        logging.info(f"[vLLM] Initializing engine from {model_name}, "
                     f"gpu_memory_utilization={gpu_memory_utilization}, "
                     f"max_model_len={max_model_len}")
        self.vllm_engine = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,  # weights change each kd_interval, skip CUDA graph
            trust_remote_code=True,
        )
        self.vllm_sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.gen_temperature,
        )
        logging.info("[vLLM] Engine initialized.")

    def _sync_weights_to_vllm(self, model):
        """Copy current student weights (float32) into vLLM engine (bfloat16)."""
        import time as _time
        _t0 = _time.time()

        vllm_model = (self.vllm_engine.llm_engine
                      .model_executor.driver_worker
                      .model_runner.model)

        vllm_state = {k: v for k, v in vllm_model.named_parameters()}
        for name, param in model.named_parameters():
            if name in vllm_state:
                vllm_state[name].data.copy_(param.data.to(vllm_state[name].dtype))

        _elapsed = _time.time() - _t0
        logging.info(f"[vLLM] Weight sync done in {_elapsed:.2f}s")

    def _get_prompt_pool(self):
        """Lazily build a list of raw prompt_ids tensors from prompt_dataset or train_dataset."""
        if self._prompt_pool is None:
            pool = []
            source = self.prompt_dataset if self.prompt_dataset is not None else self.train_dataset
            for sample in source:
                if "prompt_ids" in sample:
                    p = sample["prompt_ids"]
                    if not isinstance(p, torch.Tensor):
                        p = torch.tensor(p, dtype=torch.long)
                    pool.append(p)
            self._prompt_pool = pool
            logging.info(f"[Buffer] Prompt pool built: {len(pool)} prompts")
        return self._prompt_pool

    def _fill_rollout_buffer(self, model):
        """Sample kd_buffer_size prompts, generate as one vLLM batch, store in buffer."""
        import time as _time
        pool = self._get_prompt_pool()
        if not pool:
            logging.warning("[Buffer] Prompt pool empty, skipping buffer fill.")
            return

        n = min(self.kd_buffer_size, len(pool))
        sampled = random.sample(pool, n)

        # Left-pad to same length for vLLM batch input
        max_plen = max(p.shape[0] for p in sampled)
        pad_id = self.tokenizer.pad_token_id
        device = next(model.parameters()).device

        batch_ids = torch.full((n, max_plen), pad_id, dtype=torch.long, device=device)
        batch_mask = torch.zeros(n, max_plen, dtype=torch.long, device=device)
        prompt_lens = []
        for i, p in enumerate(sampled):
            plen = p.shape[0]
            batch_ids[i, -plen:] = p.to(device)
            batch_mask[i, -plen:] = 1
            prompt_lens.append(plen)

        _t0 = _time.time()
        generated = self._generate_with_vllm(batch_ids, batch_mask, model)
        full_mask = (generated != pad_id).long()

        self._rollout_buffer.clear()
        for i in range(generated.shape[0]):
            self._rollout_buffer.append({
                "input_ids": generated[i:i+1],
                "attention_mask": full_mask[i:i+1],
                "prompt_len": torch.tensor(prompt_lens[i]),
            })
        logging.info(f"[Buffer] Filled {len(self._rollout_buffer)} rollouts in "
                     f"{_time.time()-_t0:.1f}s at step {self.state.global_step}")

    def _generate_with_vllm(self, prompt_ids: torch.Tensor,
                             prompt_mask: torch.Tensor,
                             model) -> torch.Tensor:
        """
        Generate rollout using vLLM engine.
        Syncs student weights first (unless generate_with_teacher=True), then generates.
        Returns tensor of shape (B, prompt_len + gen_len), same as model.generate().
        """
        import time as _time
        _t0 = _time.time()

        if not self.generate_with_teacher:
            self._sync_weights_to_vllm(model)

        # Convert token IDs to list of lists (vLLM input format)
        prompt_lens = prompt_mask.sum(dim=1).tolist()
        prompts_token_ids = []
        for i, plen in enumerate(prompt_lens):
            plen = int(plen)
            prompts_token_ids.append(prompt_ids[i, -plen:].tolist())

        outputs = self.vllm_engine.generate(
            prompt_token_ids=prompts_token_ids,
            sampling_params=self.vllm_sampling_params,
        )

        # Reconstruct full sequences (prompt + generated) as padded tensor
        device = prompt_ids.device
        results = []
        for i, output in enumerate(outputs):
            gen_ids = list(output.outputs[0].token_ids)
            full_ids = prompts_token_ids[i] + gen_ids
            results.append(torch.tensor(full_ids, dtype=torch.long, device=device))

        # Right-pad to same length
        max_len = max(t.shape[0] for t in results)
        pad_id = self.tokenizer.pad_token_id
        padded = torch.full((len(results), max_len), pad_id, dtype=torch.long, device=device)
        for i, t in enumerate(results):
            padded[i, :t.shape[0]] = t

        _elapsed = _time.time() - _t0
        _gen_tokens = padded.shape[1] - int(min(prompt_lens))
        logging.info(f"[vLLM] generation done: {_gen_tokens} tokens in {_elapsed:.2f}s "
                     f"({_gen_tokens/_elapsed:.1f} tok/s)")
        return padded

    def _is_hybrid_batch(self, inputs):
        """True if batch contains NTP labels (MixedTextDataset or random CoT windows)."""
        return "labels" in inputs

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Generate on-policy completions, then run standard ADMM step."""
        # Lazy setup: install TR z_override_fn once the optimizer exists
        if getattr(self.args, 'admm_tr_z_proj', False) and not self._tr_z_initialized:
            self.setup_global_tr_z(model)
            self._tr_z_initialized = True
        # Update cal_batch for TR KL computation (used by z_override_fn callback)
        if getattr(self.args, 'admm_tr_z_proj', False):
            self._tr_current_batch = {k: v.detach() if isinstance(v, torch.Tensor) else v
                                      for k, v in inputs.items()}

        if self.offpolicy_kd and self._is_hybrid_batch(inputs):
            return self._training_step_offpolicy_kd(model, inputs, num_items_in_batch)
        elif self._is_hybrid_batch(inputs):
            return self._training_step_hybrid(model, inputs, num_items_in_batch)
        else:
            return self._training_step_kd_only(model, inputs, num_items_in_batch)

    def _training_step_kd_only(self, model, inputs, num_items_in_batch=None):
        """Original behavior: KD-only (no CoT NTP), prompt-only dataset."""
        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        prompt_len = inputs["prompt_len"].item() if inputs["prompt_len"].dim() == 0 else int(inputs["prompt_len"][0])

        # Generation (student on-policy, or teacher if generate_with_teacher=True)
        if self.use_vllm:
            generated = self._generate_with_vllm(prompt_ids, prompt_mask, model)
        else:
            gen_model = self.teacher_model if self.generate_with_teacher and self.teacher_model is not None else model
            _gc_enabled = getattr(model, "is_gradient_checkpointing", False)
            if _gc_enabled:
                model.gradient_checkpointing_disable()
            gen_model.config.use_cache = True
            gen_model.eval()
            import time as _time
            _t0 = _time.time()
            with torch.no_grad():
                generated = gen_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                )
            _elapsed = _time.time() - _t0
            _gen_tokens = generated.shape[1] - prompt_len
            logging.info(
                f"[KV-DEBUG] use_cache={gen_model.config.use_cache}, "
                f"gen_tokens={_gen_tokens}, time={_elapsed:.1f}s, "
                f"tok/s={_gen_tokens/_elapsed:.1f}"
            )
            if gen_model is model:
                model.train()
            model.config.use_cache = False
            if _gc_enabled:
                model.gradient_checkpointing_enable()

        full_mask = (generated != self.tokenizer.pad_token_id).long()
        updated_inputs = {
            "input_ids": generated,
            "attention_mask": full_mask,
            "prompt_len": inputs["prompt_len"],
        }
        self._kd_inputs = None  # use updated_inputs directly (original path)
        return super().training_step(model, updated_inputs, num_items_in_batch)

    def _training_step_offpolicy_kd(self, model, inputs, num_items_in_batch=None):
        """Off-policy KD: student+teacher forward on dataset CoT sequences. No vLLM, no buffer.

        OPD (on-policy rollout, generated by the live x model) pool refill/pop lives here too --
        _compute_loss_offpolicy_kd reads self._opd_inputs the same way
        _compute_loss_hybrid's OPD branch does, but this path has no NTP term
        (kd_lambda/opd_lambda each apply directly, not divided by 3).
        """
        cot_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
        if self.opd_enabled and self._opd_vllm_engine is not None:
            _admm_interval = self.args.admm_interval
            _opt_step = getattr(self._get_admm_optimizer(), 'current_step', self.state.global_step)
            if _opt_step % _admm_interval == 0 and _opt_step != self._opd_last_refill_step:
                self._generate_opd_pool(model)
                self._opd_last_refill_step = _opt_step
            _device = next(model.parameters()).device
            self._opd_inputs = self._pop_opd_inputs(_device)
        return super().training_step(model, cot_inputs, num_items_in_batch)

    def _training_step_hybrid(self, model, inputs, num_items_in_batch=None):
        """
        Hybrid: NTP on CoT every step + on-policy KD.

        Two KD modes:
        - Buffered (kd_buffer_size > 0): generate kd_buffer_size rollouts in one vLLM
          batch every kd_buffer_refresh_interval steps; consume one per step (kd every step).
        - Original (kd_buffer_size == 0): generate one rollout every kd_interval steps.
        """
        if self.teacher_model is None:
            self._kd_inputs = None
            self._opkd_inputs = None
        elif self.kd_triple_loss:
            # Triple loss: NTP + dataset KD + on-policy KD
            # Dataset KD: CoT batch directly (no generation)
            self._kd_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "prompt_len": inputs["prompt_len"],
            }
            # OPKD: student rollout via vLLM every kd_interval steps
            if self.state.global_step % self.kd_interval == 0:
                prompt_ids = inputs["prompt_ids"]
                prompt_mask = inputs["prompt_mask"]
                generated = self._generate_with_vllm(prompt_ids, prompt_mask, model)
                full_mask = (generated != self.tokenizer.pad_token_id).long()
                self._opkd_inputs = {
                    "input_ids": generated,
                    "attention_mask": full_mask,
                    "prompt_len": inputs["prompt_len"],
                }
                # Reuse rollout as TR-z calibration batch (richer than CoT prefix)
                if self.admm_tr_use_opkd_rollout and hasattr(self, '_tr_current_batch'):
                    self._tr_current_batch = {
                        "input_ids": generated.detach(),
                        "attention_mask": full_mask.detach(),
                    }
            else:
                self._opkd_inputs = None
        elif self.kd_offpolicy_ntp:
            # Dataset-based KD: no generation — use CoT batch directly.
            # KL(student || teacher) is computed on answer tokens (after prompt_len).
            self._kd_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "prompt_len": inputs["prompt_len"],
            }
            self._opkd_inputs = None

            # OPD: generate on-policy (x-generated) rollout pool at the start of each admm_interval block.
            # Pop one rollout per micro-step as _opd_inputs for backward KL loss.
            if self.opd_enabled and self._opd_vllm_engine is not None:
                _admm_interval = self.args.admm_interval
                _opt_step = getattr(self._get_admm_optimizer(), 'current_step', self.state.global_step)
                # Refill once per interval block (not once per micro-batch within that block).
                if _opt_step % _admm_interval == 0 and _opt_step != self._opd_last_refill_step:
                    self._generate_opd_pool(model)
                    self._opd_last_refill_step = _opt_step
                _device = next(model.parameters()).device
                self._opd_inputs = self._pop_opd_inputs(_device)
        elif self.use_vllm and self.kd_buffer_size > 0:
            # --- Buffered rollout mode ---
            # Pop once per optimizer step (global_step increments after accumulation),
            # reuse the same kd_inputs for all micro-batches within the same step.
            step = self.state.global_step
            if step != self._last_kd_step:
                if step % self.kd_step_interval == 0:
                    if len(self._rollout_buffer) == 0 or step % self.kd_buffer_refresh_interval == 0:
                        self._fill_rollout_buffer(model)
                    self._kd_inputs = self._rollout_buffer.popleft() if self._rollout_buffer else None
                else:
                    self._kd_inputs = None
                self._last_kd_step = step
        elif self.state.global_step % self.kd_interval == 0:
            # --- Original: one generation per kd_interval steps ---
            prompt_ids = inputs["prompt_ids"]
            prompt_mask = inputs["prompt_mask"]
            prompt_len = int(inputs["prompt_len"].item() if inputs["prompt_len"].dim() == 0 else inputs["prompt_len"][0])

            if self.use_vllm:
                generated = self._generate_with_vllm(prompt_ids, prompt_mask, model)
            else:
                gen_model = self.teacher_model if self.generate_with_teacher and self.teacher_model is not None else model
                _gc_enabled = getattr(model, "is_gradient_checkpointing", False)
                if _gc_enabled:
                    model.gradient_checkpointing_disable()
                gen_model.config.use_cache = True
                gen_model.eval()
                import time as _time
                _t0 = _time.time()
                with torch.no_grad():
                    generated = gen_model.generate(
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        generation_config=self.generation_config,
                    )
                _elapsed = _time.time() - _t0
                _gen_tokens = generated.shape[1] - prompt_len
                logging.info(
                    f"[KD] step={self.state.global_step}, "
                    f"gen_tokens={_gen_tokens}, time={_elapsed:.1f}s"
                )
                if gen_model is model:
                    model.train()
                model.config.use_cache = False
                if _gc_enabled:
                    model.gradient_checkpointing_enable()

            full_mask = (generated != self.tokenizer.pad_token_id).long()
            self._kd_inputs = {
                "input_ids": generated,
                "attention_mask": full_mask,
                "prompt_len": inputs["prompt_len"],
            }
        else:
            self._kd_inputs = None

        # Pass CoT NTP inputs to compute_loss via super()
        ntp_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
        return super().training_step(model, ntp_inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Three modes:
        1. Off-policy KD (labels in inputs, offpolicy_kd=True): KL on dataset CoT tokens, no NTP
        2. Hybrid (labels in inputs): NTP on CoT always + KD loss when self._kd_inputs is set
        3. KD-only (no labels): Reverse KL(student || teacher) on generated tokens only
        """
        if self.offpolicy_kd and "labels" in inputs:
            return self._compute_loss_offpolicy_kd(model, inputs, return_outputs)
        elif "labels" in inputs:
            return self._compute_loss_hybrid(model, inputs, return_outputs)
        else:
            return self._compute_loss_kd_only(model, inputs, return_outputs)

    def _compute_loss_offpolicy_kd(self, model, inputs, return_outputs=False):
        """KL(student || teacher) on dataset CoT answer tokens. No NTP, no vLLM."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # Derive answer boundary from labels (-100 = problem tokens or padding)
        answer_pos = (labels[0] != -100).nonzero(as_tuple=False)
        if len(answer_pos) == 0:
            student_out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return (loss, student_out) if return_outputs else loss

        prompt_len = answer_pos[0].item()
        gen_len = len(answer_pos)

        student_out = model(input_ids=input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            teacher_out = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)

        kd_loss, opd_metrics = self._kl_loss(
            student_out.logits, teacher_out.logits,
            attention_mask, prompt_len, gen_len,
        )
        log_dict = {"train/offpolicy_kd_loss": kd_loss.item()}
        log_dict.update({k: v.item() for k, v in opd_metrics.items()})

        # KD(dataset, reverse-KL-ish per _kl_loss) + OPD (on-policy, x-generated
        # rollout, reverse KL) -- no NTP term here (unlike _compute_loss_hybrid's
        # OPD branch, which always mixes in ntp_loss at opd_lambda/3). kd_lambda
        # and opd_lambda are each used directly as that loss term's own weight
        # (e.g. 0.5/0.5), not divided by 3.
        if self.opd_enabled and self._opd_inputs is not None and self.opd_lambda > 0:
            opd_loss, opd_bkwd_metrics = self._compute_opd_backward_kl(model)
            loss = self.kd_lambda * kd_loss + self.opd_lambda * opd_loss
            log_dict["train/opd_loss"] = opd_loss.item()
            log_dict.update({f"opd/{k.split('/')[-1]}": v.item() for k, v in opd_bkwd_metrics.items()})
        else:
            loss = self.kd_lambda * kd_loss

        self.log(log_dict)
        return (loss, student_out) if return_outputs else loss

    def _compute_loss_hybrid(self, model, inputs, return_outputs=False):
        """NTP on CoT + optional KD loss."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        # Student forward on CoT sequence
        student_out = model(input_ids=input_ids, attention_mask=attention_mask)

        # NTP loss: cross-entropy on CoT tokens (problem tokens masked with -100 in labels)
        shift_logits = student_out.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ntp_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )

        if self.kd_triple_loss:
            # NTP(ntp_lambda) + dataset KD(kd_lambda) + OPKD(kd_opkd_lambda)
            loss = self.ntp_lambda * ntp_loss
            log_dict = {"train/ntp_loss": ntp_loss.item()}
            if self._kd_inputs is not None:
                kd_loss, _ = self._compute_kd_forward(model)
                loss = loss + self.kd_lambda * kd_loss
                log_dict["train/kd_loss"] = kd_loss.item()
            if self._opkd_inputs is not None:
                opkd_loss, opd_metrics = self._compute_kd_forward(model, kd_inputs=self._opkd_inputs)
                loss = loss + self.kd_opkd_lambda * opkd_loss
                log_dict["train/opkd_loss"] = opkd_loss.item()
                log_dict.update({k: v.item() for k, v in opd_metrics.items()})
            self.log(log_dict)
            if wandb.run is not None:
                wandb.log({k: v for k, v in log_dict.items()}, commit=False)
        elif self._kd_inputs is not None:
            kd_loss, opd_metrics = self._compute_kd_forward(model)
            log_dict = {"train/ntp_loss": ntp_loss.item(), "train/kd_loss": kd_loss.item()}
            log_dict.update({k: v.item() for k, v in opd_metrics.items()})

            # OPD: backward KL D(student||teacher) on x-generated rollout tokens
            if self.opd_enabled and self._opd_inputs is not None and self.opd_lambda > 0:
                opd_loss, opd_bkwd_metrics = self._compute_opd_backward_kl(model)
                ntp_w = kd_w = opd_w = self.opd_lambda / 3.0
                loss = ntp_w * ntp_loss + kd_w * kd_loss + opd_w * opd_loss
                log_dict["train/opd_loss"] = opd_loss.item()
                log_dict.update({f"opd/{k.split('/')[-1]}": v.item()
                                 for k, v in opd_bkwd_metrics.items()})
            else:
                loss = ntp_loss + self.kd_lambda * kd_loss

            self.log(log_dict)
            if wandb.run is not None:
                wandb.log({"train/kd_loss": kd_loss.item()}, commit=False)
        else:
            loss = ntp_loss
            self.log({"train/ntp_loss": ntp_loss.item()})

        return (loss, student_out) if return_outputs else loss

    def _compute_loss_kd_only(self, model, inputs, return_outputs=False, **kwargs):
        """Original KD-only path: Reverse KL(student || teacher) on generated tokens."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_len = inputs["prompt_len"].item() if inputs["prompt_len"].dim() == 0 else int(inputs["prompt_len"][0])

        gen_len = input_ids.shape[1] - prompt_len
        if gen_len <= 0:
            logging.warning("No generated tokens found; skipping KD loss.")
            student_out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            return (loss, student_out) if return_outputs else loss

        # Student forward
        student_out = model(input_ids=input_ids, attention_mask=attention_mask)

        # Teacher forward (frozen, no grad)
        with torch.no_grad():
            teacher_out = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        kd_loss, _ = self._kl_loss(
            student_out.logits, teacher_out.logits,
            attention_mask, prompt_len, gen_len,
        )

        # NTP loss on prompt tokens (optional, controlled by ntp_lambda)
        if self.ntp_lambda > 0.0 and prompt_len > 1:
            ntp_logits = student_out.logits[:, :prompt_len - 1, :]
            ntp_labels = input_ids[:, 1:prompt_len]
            ntp_mask = attention_mask[:, 1:prompt_len].float()
            ntp_loss_per_tok = F.cross_entropy(
                ntp_logits.reshape(-1, ntp_logits.shape[-1]),
                ntp_labels.reshape(-1),
                reduction='none',
            ).reshape(ntp_labels.shape)
            ntp_loss = (ntp_loss_per_tok * ntp_mask).sum() / ntp_mask.sum().clamp(min=1)
            loss = kd_loss + self.ntp_lambda * ntp_loss
        else:
            loss = kd_loss

        return (loss, student_out) if return_outputs else loss

    def _compute_opd_backward_kl(self, model):
        """Backward KL D(student||teacher) on OPD rollout tokens. Uses compute_self_distillation_loss (alpha=1.0)."""
        from argparse import Namespace
        kd = self._opd_inputs
        input_ids = kd["input_ids"]
        attention_mask = kd["attention_mask"]
        prompt_len = int(kd["prompt_len"].item() if kd["prompt_len"].dim() == 0 else kd["prompt_len"][0])
        gen_len = input_ids.shape[1] - prompt_len
        if gen_len <= 0:
            logging.warning("OPD: no generated tokens in rollout, skipping.")
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True), {}

        student_out = model(input_ids=input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            teacher_out = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)

        curr_gen_len = gen_len
        student_all_logps = F.log_softmax(
            student_out.logits[:, prompt_len - 1: prompt_len - 1 + curr_gen_len, :] / self.kd_temperature, dim=-1
        )
        teacher_all_logps = F.log_softmax(
            teacher_out.logits[:, prompt_len - 1: prompt_len - 1 + curr_gen_len, :] / self.kd_temperature, dim=-1
        )
        response_mask = attention_mask[:, prompt_len: prompt_len + curr_gen_len].float()

        distill_cfg = Namespace(full_logit_distillation=True, distillation_topk=None,
                                distillation_add_tail=False, alpha=1.0, is_clip=None)
        loss, _ = compute_self_distillation_loss(
            student_log_probs=None, teacher_log_probs=None,
            response_mask=response_mask,
            self_distillation_config=distill_cfg,
            student_all_log_probs=student_all_logps,
            teacher_all_log_probs=teacher_all_logps,
            loss_agg_mode="token-mean",
        )
        return loss, {}

    def _compute_kd_forward(self, model, kd_inputs=None):
        """Run student+teacher on kd_inputs (defaults to self._kd_inputs), return KD loss and OPD metrics."""
        kd = kd_inputs if kd_inputs is not None else self._kd_inputs
        input_ids = kd["input_ids"]
        attention_mask = kd["attention_mask"]
        prompt_len = int(kd["prompt_len"].item() if kd["prompt_len"].dim() == 0 else kd["prompt_len"][0])
        gen_len = input_ids.shape[1] - prompt_len

        if gen_len <= 0:
            logging.warning("KD: no generated tokens, skipping.")
            return torch.tensor(0.0, device=input_ids.device), {}

        student_out = model(input_ids=input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            teacher_out = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)

        return self._kl_loss(
            student_out.logits, teacher_out.logits,
            attention_mask, prompt_len, gen_len,
        )

    def _kl_loss(self, s_logits, t_logits, attention_mask, prompt_len, gen_len):
        """Reverse KL(student || teacher) on generated tokens. Returns (loss, opd_metrics).

        prompt_len==0 (e.g. FineWeb-Edu pretrain samples, which have no masked
        prompt prefix — the whole sequence is "answer") used to make the
        `prompt_len - 1 : -1` slice literally empty (Python's `-1:-1` is
        always zero-length, not "from -1 to the end"), crashing downstream on
        a size-0 vs. size-gen_len mismatch. Position 0 has no preceding
        logit to predict it from regardless (standard autoregressive
        next-token shift), so when prompt_len==0 the valid generation region
        starts at logit index 0 (predicting token index 1) same as any other
        next-token loss — clamp the slice start at 0 and derive gen_mask's
        length from the actual slice instead of the passed-in gen_len, which
        also makes this robust to any other prompt_len/gen_len mismatch.
        """
        start = max(prompt_len - 1, 0)
        s_logits_gen = s_logits[:, start: -1, :]   # (B, L, V)
        t_logits_gen = t_logits[:, start: -1, :]
        mask_start = start + 1
        gen_mask = attention_mask[:, mask_start: mask_start + s_logits_gen.size(1)].float()
        opd_metrics = {}

        if self.kd_topk > 0:
            s_logp_full = F.log_softmax(s_logits_gen / self.kd_temperature, dim=-1)
            t_logp_full = F.log_softmax(t_logits_gen / self.kd_temperature, dim=-1)

            if self.forward_kl:
                # Forward KL: D(teacher || student) = E_teacher[log t/s], gather at teacher top-K
                t_topk_idx = t_logits_gen.topk(self.kd_topk, dim=-1).indices
                s_logp = s_logp_full.gather(-1, t_topk_idx)
                t_logp = t_logp_full.gather(-1, t_topk_idx)
                kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
            else:
                # Reverse KL: D(student || teacher), gather at student top-K
                s_topk = s_logits_gen.topk(self.kd_topk, dim=-1)
                s_topk_idx = s_topk.indices                          # (B, gen_len, K)
                s_logp = s_logp_full.gather(-1, s_topk_idx)
                t_logp = t_logp_full.gather(-1, s_topk_idx)
                kl = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)
            loss = (kl * gen_mask).sum() / gen_mask.sum().clamp(min=1)

            with torch.no_grad():
                t_topk = t_logits_gen.topk(self.kd_topk, dim=-1)
                t_topk_idx = t_topk.indices
                # For forward KL, s_topk_idx was not computed above — compute it here for metrics
                if self.forward_kl:
                    s_topk_idx = s_logits_gen.topk(self.kd_topk, dim=-1).indices

                # Overlap Ratio: |S_t ∩ T_t| / K
                overlap_mask = (s_topk_idx.unsqueeze(-1) == t_topk_idx.unsqueeze(-2)).any(dim=-1)
                overlap_ratio = overlap_mask.float().mean(dim=-1)
                opd_metrics["kd/overlap_ratio"] = (overlap_ratio * gen_mask).sum() / gen_mask.sum().clamp(min=1)

                # Entropy Gap: H(T_top-K) - H(S_top-K) approximation
                s_ent = -(s_logp.exp() * s_logp).sum(dim=-1)
                t_logp_topk = t_logp_full.gather(-1, t_topk_idx)
                t_ent = -(t_logp_topk.exp() * t_logp_topk).sum(dim=-1)
                opd_metrics["kd/entropy_gap"] = ((s_ent - t_ent).abs() * gen_mask).sum() / gen_mask.sum().clamp(min=1)

                # Overlap-Token Advantage: within intersection, compute KL advantage
                inter_mask = overlap_mask.float()
                s_logp_inter = s_logp.masked_fill(inter_mask == 0, float('-inf'))
                t_logp_inter = t_logp.masked_fill(inter_mask == 0, float('-inf'))
                s_logp_inter_norm = s_logp_inter - torch.logsumexp(s_logp_inter, dim=-1, keepdim=True)
                t_logp_inter_norm = t_logp_inter - torch.logsumexp(t_logp_inter, dim=-1, keepdim=True)
                adv = (s_logp_inter_norm.exp() * (t_logp_inter_norm - s_logp_inter_norm)).sum(dim=-1)
                has_overlap = (inter_mask.sum(dim=-1) > 0).float()
                adv = torch.nan_to_num(adv, nan=0.0)
                opd_metrics["kd/overlap_token_advantage"] = (adv * has_overlap * gen_mask).sum() / (has_overlap * gen_mask).sum().clamp(min=1)
        else:
            s_logp = F.log_softmax(s_logits_gen / self.kd_temperature, dim=-1)
            t_logp = F.log_softmax(t_logits_gen / self.kd_temperature, dim=-1)
            if self.forward_kl:
                kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
            else:
                kl = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)
            loss = (kl * gen_mask).sum() / gen_mask.sum().clamp(min=1)

            # Diagnostic-only top-100 overlap ratio (independent of kd_topk=0's
            # full-vocab loss above -- purely for logging, no gradient, fixed
            # K=100 regardless of what kd_topk the actual loss uses).
            with torch.no_grad():
                _diag_k = min(100, s_logits_gen.size(-1))
                s_topk_idx = s_logits_gen.topk(_diag_k, dim=-1).indices
                t_topk_idx = t_logits_gen.topk(_diag_k, dim=-1).indices
                overlap_mask = (s_topk_idx.unsqueeze(-1) == t_topk_idx.unsqueeze(-2)).any(dim=-1)
                overlap_ratio = overlap_mask.float().mean(dim=-1)
                opd_metrics["kd/overlap_ratio_top100"] = (
                    (overlap_ratio * gen_mask).sum() / gen_mask.sum().clamp(min=1)
                )

        return loss, opd_metrics
