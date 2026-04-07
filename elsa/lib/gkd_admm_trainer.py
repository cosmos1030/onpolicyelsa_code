"""
GKDADMMTrainer: ADMM pruning with on-policy knowledge distillation loss.
Inherits ADMMTrainer to keep ADMM mechanics intact, replaces NTP loss with KD.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import GenerationConfig
from .trainer import ADMMTrainer
from absl import logging
import json
import random


# ---------------------------------------------------------------------------
# Dataset: prompt-only from math 220k JSONL
# ---------------------------------------------------------------------------
class MathPromptDataset(Dataset):
    """
    Loads math prompts from a JSONL file (uses 'prompt' field, chat-template applied).
    Returns tokenized prompt tensors for on-policy generation.
    """
    def __init__(self, jsonl_path, tokenizer, max_prompt_len=512, nsamples=None, seed=42):
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

        logging.info(f"MathPromptDataset: {len(self.samples)} prompts loaded from {jsonl_path}")

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

        self.generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=gen_temperature,
            use_cache=True,  # generation is under no_grad, KV cache is safe
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Move teacher to same device as student; keep frozen
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Generate on-policy completions, then run standard ADMM step."""
        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        prompt_len = inputs["prompt_len"].item() if inputs["prompt_len"].dim() == 0 else int(inputs["prompt_len"][0])

        # 1. On-policy generation (student, no grad)
        # Temporarily enable use_cache and disable gradient checkpointing for fast generation
        _gc_enabled = getattr(model, "is_gradient_checkpointing", False)
        if _gc_enabled:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        model.eval()
        import time as _time
        _t0 = _time.time()
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
            )
        _elapsed = _time.time() - _t0
        _gen_tokens = generated.shape[1] - prompt_len
        logging.info(
            f"[KV-DEBUG] use_cache={model.config.use_cache}, "
            f"gen_tokens={_gen_tokens}, time={_elapsed:.1f}s, "
            f"tok/s={_gen_tokens/_elapsed:.1f}"
        )
        model.train()
        model.config.use_cache = False
        if _gc_enabled:
            model.gradient_checkpointing_enable()

        # 2. Build new inputs with full (prompt + completion) sequence
        full_mask = (generated != self.tokenizer.pad_token_id).long()
        updated_inputs = {
            "input_ids": generated,
            "attention_mask": full_mask,
            "prompt_len": inputs["prompt_len"],
        }

        return super().training_step(model, updated_inputs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Reverse KL(student || teacher) on generated tokens only."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_len = inputs["prompt_len"].item() if inputs["prompt_len"].dim() == 0 else int(inputs["prompt_len"][0])

        gen_len = input_ids.shape[1] - prompt_len
        if gen_len <= 0:
            logging.warning("No generated tokens found; skipping KD loss.")
            return super().compute_loss(model, inputs, return_outputs)

        # Student forward
        student_out = model(input_ids=input_ids, attention_mask=attention_mask)

        # Teacher forward (frozen, no grad)
        with torch.no_grad():
            teacher_out = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Logit at position t predicts token t+1.
        # Generated tokens start at prompt_len, so we need logits [prompt_len-1 : -1].
        s_logits = student_out.logits[:, prompt_len - 1: -1, :]   # (B, gen_len, V)
        t_logits = teacher_out.logits[:, prompt_len - 1: -1, :]   # (B, gen_len, V)

        # Compute full-vocab log-probs first, then restrict to teacher's top-k
        s_logp_full = F.log_softmax(s_logits / self.kd_temperature, dim=-1)
        t_logp_full = F.log_softmax(t_logits / self.kd_temperature, dim=-1)

        if self.kd_topk > 0:
            topk_idx = t_logits.topk(self.kd_topk, dim=-1).indices
            s_logp = s_logp_full.gather(-1, topk_idx)
            t_logp = t_logp_full.gather(-1, topk_idx)
        else:
            s_logp = s_logp_full
            t_logp = t_logp_full

        # Reverse KL: KL(student || teacher) = sum p_s * (log p_s - log p_t)
        gen_mask = attention_mask[:, prompt_len: prompt_len + gen_len].float()
        kl = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)  # (B, gen_len)
        kd_loss = (kl * gen_mask).sum() / gen_mask.sum().clamp(min=1)

        # NTP loss on prompt tokens (optional, controlled by ntp_lambda)
        if self.ntp_lambda > 0.0 and prompt_len > 1:
            ntp_logits = student_out.logits[:, :prompt_len - 1, :]  # predicts tokens 1..prompt_len
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
