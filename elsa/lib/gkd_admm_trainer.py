"""
GKDADMMTrainer: ADMM pruning with on-policy knowledge distillation loss.
Inherits ADMMTrainer to keep ADMM mechanics intact, replaces NTP loss with KD.
"""
import collections
import hashlib
import os
import pickle
import random

import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import GenerationConfig
from .trainer import ADMMTrainer
from absl import logging
import json


def _dataset_cache_path(cache_dir, jsonl_path, tokenizer_name, **kwargs):
    key = f"{jsonl_path}|{tokenizer_name}|" + "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{h}.pkl")


# ---------------------------------------------------------------------------
# Dataset: prompt-only from math 220k JSONL
# ---------------------------------------------------------------------------
class MathPromptDataset(Dataset):
    """
    Loads math prompts from a JSONL file (uses 'prompt' field, chat-template applied).
    Returns tokenized prompt tensors for on-policy generation.
    """
    def __init__(self, jsonl_path, tokenizer, max_prompt_len=512, nsamples=None, seed=42,
                 cache_dir="/tmp/elsa_dataset_cache"):
        cache_path = _dataset_cache_path(cache_dir, jsonl_path, tokenizer.name_or_path,
                                         cls="MathPrompt", max_prompt_len=max_prompt_len,
                                         nsamples=nsamples, seed=seed)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)
            logging.info(f"MathPromptDataset: loaded {len(self.samples)} samples from cache {cache_path}")
            return

        random.seed(seed)
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        if nsamples and nsamples < len(records):
            records = random.sample(records, nsamples)

        self.samples = []
        for rec in records:
            prompt = rec.get("prompt", "")
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

        with open(cache_path, "wb") as f:
            pickle.dump(self.samples, f)
        logging.info(f"MathPromptDataset: {len(self.samples)} prompts loaded and cached to {cache_path}")

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
class MathCotKDDataset(Dataset):
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

    def __init__(self, jsonl_path, tokenizer, max_len=2048, max_prompt_len=512,
                 nsamples=None, seed=42, cache_dir="/tmp/elsa_dataset_cache"):
        cache_path = _dataset_cache_path(cache_dir, jsonl_path, tokenizer.name_or_path,
                                         cls="MathCotKD", max_len=max_len,
                                         max_prompt_len=max_prompt_len,
                                         nsamples=nsamples, seed=seed)
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.samples = pickle.load(f)
            logging.info(f"MathCotKDDataset: loaded {len(self.samples)} samples from cache {cache_path}")
            return

        random.seed(seed)
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        if nsamples and nsamples < len(records):
            records = random.sample(records, nsamples)

        self.samples = []
        for rec in records:
            text = rec.get("text", "")
            if not text:
                continue

            # Split at <think>
            idx = text.find(self.THINK_TAG)
            if idx == -1:
                # fallback: split at first double-newline
                idx = text.find("\n\n")
                if idx == -1:
                    continue
                prompt_text = text[:idx]
                cot_text = text[idx + 2:]
            else:
                prompt_text = text[:idx + len(self.THINK_TAG)]
                cot_text = text[idx:]  # includes <think>CoT</think>answer

            if not cot_text.strip():
                continue

            # Full sequence for NTP
            full_enc = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                padding=False,
            )
            full_ids = full_enc["input_ids"].squeeze(0)
            full_mask = full_enc["attention_mask"].squeeze(0)

            # Prompt for KD generation
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

            # Labels: mask problem tokens with -100, keep CoT tokens
            labels = full_ids.clone()
            labels[:prompt_len] = -100

            self.samples.append({
                "input_ids": full_ids,
                "attention_mask": full_mask,
                "labels": labels,
                "prompt_ids": prompt_ids,
                "prompt_mask": prompt_mask_t,
            })

        with open(cache_path, "wb") as f:
            pickle.dump(self.samples, f)
        logging.info(f"MathCotKDDataset: {len(self.samples)} samples loaded and cached to {cache_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_cot_kd(pad_token_id):
    """
    Collate for MathCotKDDataset.
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
        generate_with_teacher: bool = False,
        forward_kl: bool = False,
        prompt_dataset=None,
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
        """True if batch contains NTP labels (MathCotKDDataset or random CoT windows)."""
        return "labels" in inputs

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Generate on-policy completions, then run standard ADMM step."""
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
        """Off-policy KD: student+teacher forward on dataset CoT sequences. No vLLM, no buffer."""
        cot_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
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
        loss = self.kd_lambda * kd_loss
        log_dict = {"train/offpolicy_kd_loss": kd_loss.item()}
        log_dict.update({k: v.item() for k, v in opd_metrics.items()})
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

        if self._kd_inputs is not None:
            kd_loss, opd_metrics = self._compute_kd_forward(model)
            loss = ntp_loss + self.kd_lambda * kd_loss
            log_dict = {"train/ntp_loss": ntp_loss.item(), "train/kd_loss": kd_loss.item()}
            log_dict.update({k: v.item() for k, v in opd_metrics.items()})
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

    def _compute_kd_forward(self, model):
        """Run student+teacher on self._kd_inputs, return KD loss and OPD metrics."""
        kd = self._kd_inputs
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
        """Reverse KL(student || teacher) on generated tokens. Returns (loss, opd_metrics)."""
        s_logits_gen = s_logits[:, prompt_len - 1: -1, :]   # (B, gen_len, V)
        t_logits_gen = t_logits[:, prompt_len - 1: -1, :]
        gen_mask = attention_mask[:, prompt_len: prompt_len + gen_len].float()
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

        return loss, opd_metrics
