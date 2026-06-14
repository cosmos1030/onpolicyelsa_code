"""
GMP + DPO training utilities.  (v2 — fixed ref/EOS/logprob-scale)

Pair structure:
  chosen   = dense model continuation (pre-generated cache)
  rejected = sparse model continuation (regenerated every mask_interval steps)

v2 fixes vs v1:
  1. EOS is now allowed in generation (eos_token_id=model default).
     attention_mask reflects actual token length, not all-ones.
  2. Per-token average logprob throughout (sum / n_tokens), not raw sum.
     Eliminates beta×512-token scale blowup.
  3. ref_model is updated every mask_interval in gmp_trainer (or use
     reference_free=True to skip ref entirely).

Core DPO functions (selective_log_softmax, pad_to_length, concatenated_inputs,
concatenated_forward, dpo_loss) are ported from TRL's DPOTrainer with minimal
changes: self.* replaced by function parameters, accelerate/vision/padding_free
paths kept intact for future FSDP use.

Reference:
  RAC/open-r1-main/src/open_r1/open_r1_trl/trl/trainer/dpo_trainer.py
  RAC/open-r1-main/src/open_r1/open_r1_trl/trl/trainer/utils.py
"""

import hashlib
import math
import os
from collections import deque
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import logging
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Utilities (ported from trl/trainer/utils.py)
# ---------------------------------------------------------------------------

def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    pad_size = list(tensor.shape)
    pad_size[dim] = length - tensor.size(dim)
    return torch.cat(
        [tensor, pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device)],
        dim=dim,
    )


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Memory-efficient log_softmax -> gather. Ported from TRL utils.py."""
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        return selected_logits - logsumexp_values
    else:
        # bfloat16: logsumexp unstable, fall back to row-wise loop
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            per_token_logps.append(row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1))
        return torch.stack(per_token_logps)


# ---------------------------------------------------------------------------
# Batch construction (ported from DPOTrainer.concatenated_inputs)
# ---------------------------------------------------------------------------

def concatenated_inputs(
    batch: dict[str, torch.LongTensor],
    padding_value: int,
) -> dict[str, torch.LongTensor]:
    """
    Concatenate chosen and rejected completions into a single 2N batch.
    batch keys: prompt_input_ids, prompt_attention_mask,
                chosen_input_ids, chosen_attention_mask,
                rejected_input_ids, rejected_attention_mask
    """
    output = {}
    output["prompt_input_ids"] = torch.cat([batch["prompt_input_ids"], batch["prompt_input_ids"]], dim=0)
    output["prompt_attention_mask"] = torch.cat(
        [batch["prompt_attention_mask"], batch["prompt_attention_mask"]], dim=0
    )

    max_completion_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
    output["completion_input_ids"] = torch.cat(
        (
            pad_to_length(batch["chosen_input_ids"],   max_completion_length, pad_value=padding_value),
            pad_to_length(batch["rejected_input_ids"], max_completion_length, pad_value=padding_value),
        ),
    )
    output["completion_attention_mask"] = torch.cat(
        (
            pad_to_length(batch["chosen_attention_mask"],   max_completion_length, pad_value=0),
            pad_to_length(batch["rejected_attention_mask"], max_completion_length, pad_value=0),
        ),
    )
    return output


# ---------------------------------------------------------------------------
# Forward pass (ported from DPOTrainer.concatenated_forward)
# ---------------------------------------------------------------------------

def concatenated_forward(
    model: nn.Module,
    batch: dict[str, torch.LongTensor],
    padding_value: int,
    label_pad_token_id: int = -100,
    max_length: int = None,
    truncation_mode: str = "keep_end",
    loss_type: str = "sigmoid",
    use_logits_to_keep: bool = False,
    is_encoder_decoder: bool = False,
    is_ref_model: bool = False,
    ld_alpha: float = None,
) -> dict[str, torch.Tensor]:
    """
    Run the model on concatenated chosen+rejected inputs (one forward pass).
    Returns dict with: chosen_logps, rejected_logps, mean_chosen_logits,
    mean_rejected_logits, (optionally nll_loss).

    Ported from TRL DPOTrainer.concatenated_forward. FSDP/padding_free paths
    preserved for future use; vision-model paths stripped.
    """
    num_examples = batch["prompt_input_ids"].shape[0]
    concatenated_batch = concatenated_inputs(batch, padding_value=padding_value)

    model_kwargs = {"use_cache": False}

    prompt_input_ids      = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids  = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]

    if is_encoder_decoder:
        labels = completion_input_ids.clone()
        labels[completion_attention_mask == 0] = label_pad_token_id
        outputs = model(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            labels=labels,
            **model_kwargs,
        )
        logits = outputs.logits
        loss_mask = completion_attention_mask.bool()
    else:
        input_ids      = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask      = torch.cat(
            (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1
        )

        # Truncation
        if max_length is not None and max_length < attention_mask.size(1):
            if truncation_mode == "keep_start":
                # flush left then truncate right
                attention_mask, input_ids, loss_mask = _flush_left(attention_mask, input_ids, loss_mask)
                attention_mask = attention_mask[:, :max_length]
                input_ids      = input_ids[:, :max_length]
                loss_mask      = loss_mask[:, :max_length]
            elif truncation_mode == "keep_end":
                # flush right, truncate left, flush left
                attention_mask, input_ids, loss_mask = _flush_right(attention_mask, input_ids, loss_mask)
                input_ids      = input_ids[:, -max_length:]
                attention_mask = attention_mask[:, -max_length:]
                loss_mask      = loss_mask[:, -max_length:]
                attention_mask, input_ids, loss_mask = _flush_left(attention_mask, input_ids, loss_mask)
            else:
                raise ValueError(f"Unknown truncation_mode: {truncation_mode}")
        else:
            attention_mask, input_ids, loss_mask = _flush_left(attention_mask, input_ids, loss_mask)

        if use_logits_to_keep:
            first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
            logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
            model_kwargs["logits_to_keep"] = logits_to_keep

        model_kwargs["attention_mask"] = attention_mask
        outputs = model(input_ids, **model_kwargs)
        logits  = outputs.logits

        labels    = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if use_logits_to_keep:
            labels    = labels[:, -logits_to_keep:]
            loss_mask = loss_mask[:, -logits_to_keep:]

    if logits.shape[:2] != labels.shape[:2]:
        seq_len = labels.shape[1]
        logits  = logits[:, -seq_len:]

    # Compute per-token log-probs
    labels[~loss_mask] = 0  # dummy; ignored via loss_mask
    per_token_logps = selective_log_softmax(logits, labels)
    per_token_logps[~loss_mask] = 0
    per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)

    # FIX v2: always use per-token AVERAGE logprob (not sum).
    # With sum over 512 tokens, beta * margin blows up → sigmoid saturation.
    # Average logprob keeps the scale independent of sequence length.
    seq_lens = loss_mask.sum(-1).float().clamp(min=1)
    all_logps = per_token_logps.sum(-1) / seq_lens

    output = {}
    output["seq_lens"] = seq_lens  # expose for diagnostics

    if ld_alpha is not None and not is_ref_model:
        completion_lengths = loss_mask.sum(dim=1)
        chosen_lengths   = completion_lengths[:num_examples]
        rejected_lengths = completion_lengths[num_examples:]
        public_lengths   = torch.min(chosen_lengths, rejected_lengths)
        public_lengths   = torch.cat([public_lengths, public_lengths], dim=0)

        # Compute completion-relative positions.
        # After flush_left + roll(+1), per_token_logps has completion logprobs at
        # absolute positions [prompt_len .. prompt_len + completion_len - 1].
        # Use attention_mask to recover prompt_len per sample.
        total_lens   = model_kwargs["attention_mask"].sum(dim=1)          # prompt + completion
        prompt_lens  = (total_lens - completion_lengths).unsqueeze(1)     # [2N, 1]

        seq_len      = per_token_logps.size(1)
        position_ids = torch.arange(seq_len, device=per_token_logps.device).unsqueeze(0)  # [1, L]
        comp_rel     = position_ids - prompt_lens                          # [2N, L], 0-indexed within completion

        front_mask = (comp_rel >= 0) & (comp_rel < public_lengths.unsqueeze(1))
        rear_mask  = (comp_rel >= public_lengths.unsqueeze(1)) & (comp_rel < completion_lengths.unsqueeze(1))
        # Divide by seq_lens for length-consistent scaling (same as normal path).
        all_logps  = (
            (per_token_logps * front_mask.float()).sum(1)
            + ld_alpha * (per_token_logps * rear_mask.float()).sum(1)
        ) / seq_lens

    output["chosen_logps"]  = all_logps[:num_examples]
    output["rejected_logps"] = all_logps[num_examples:]

    if not is_encoder_decoder:
        output["mean_chosen_logits"]   = logits[:num_examples][loss_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][loss_mask[num_examples:]].mean()

    return output


def _flush_left(attention_mask, *tensors):
    """Shift all sequences to be left-aligned (remove leading padding)."""
    new_tensors = [attention_mask]
    new_tensors.extend(tensors)
    results = []
    for t in new_tensors:
        new_t = torch.zeros_like(t)
        for i, row in enumerate(t):
            non_pad = row[row != 0] if t is attention_mask else row[attention_mask[i] != 0]
            new_t[i, :len(non_pad)] = non_pad
        results.append(new_t)
    return tuple(results)


def _flush_right(attention_mask, *tensors):
    """Shift all sequences to be right-aligned (remove trailing padding)."""
    new_tensors = [attention_mask]
    new_tensors.extend(tensors)
    results = []
    for t in new_tensors:
        new_t = torch.zeros_like(t)
        for i, row in enumerate(t):
            non_pad = row[row != 0] if t is attention_mask else row[attention_mask[i] != 0]
            new_t[i, -len(non_pad):] = non_pad
        results.append(new_t)
    return tuple(results)


# ---------------------------------------------------------------------------
# DPO loss (ported from DPOTrainer.dpo_loss)
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_completion_token_logps(
    model: nn.Module,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    completion_input_ids: torch.Tensor,
    completion_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token log-probs for completion tokens only. Returns [B, comp_len].

    logits[:, prompt_len-1+k, :] predicts completion_input_ids[:, k],
    so we extract those logits and gather the actual token logprobs.
    """
    input_ids = torch.cat([prompt_input_ids, completion_input_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, completion_attention_mask], dim=1)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits.float()
    prompt_len = prompt_input_ids.shape[1]
    comp_len = completion_input_ids.shape[1]
    comp_logits = logits[:, prompt_len - 1: prompt_len - 1 + comp_len, :]
    log_probs = F.log_softmax(comp_logits, dim=-1)
    token_logps = log_probs.gather(-1, completion_input_ids.unsqueeze(-1)).squeeze(-1)
    return token_logps * completion_attention_mask.float()


def ca_ipo_loss(
    policy_chosen_token_logps: torch.Tensor,
    policy_rejected_token_logps: torch.Tensor,
    ref_chosen_token_logps: torch.Tensor,
    ref_rejected_token_logps: torch.Tensor,
    teacher_chosen_token_logps: torch.Tensor,
    teacher_rejected_token_logps: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    ref_chosen_logps_avg: torch.Tensor,
    ref_rejected_logps_avg: torch.Tensor,
    policy_chosen_logps_avg: torch.Tensor,
    policy_rejected_logps_avg: torch.Tensor,
    beta: float = 1.0,
    eps_credit: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """CA-IPO loss: token-credit-weighted IPO surrogate.

    delta (IPO residual) uses the same sequence-average log-ratios as
    existing IPO for a fair comparison. Only the gradient allocation
    (delta_w) is reweighted by teacher-student discrepancy.

    Stop-gradients: e, w_pos, w_neg are all detached.
    """
    chosen_mask  = chosen_mask.float()
    rejected_mask = rejected_mask.float()

    # token-level log-ratios for policy vs ref
    z_s_pos = (policy_chosen_token_logps  - ref_chosen_token_logps)  * chosen_mask
    z_s_neg = (policy_rejected_token_logps - ref_rejected_token_logps) * rejected_mask

    # credit weights: teacher vs student discrepancy (ref cancels)
    with torch.no_grad():
        a_pos = torch.relu(teacher_chosen_token_logps  - policy_chosen_token_logps.detach())  * chosen_mask
        a_neg = torch.relu(policy_rejected_token_logps.detach() - teacher_rejected_token_logps) * rejected_mask

        a_pos = a_pos + eps_credit * chosen_mask
        a_neg = a_neg + eps_credit * rejected_mask

        pos_count = chosen_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neg_count = rejected_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        w_pos = a_pos / a_pos.sum(dim=-1, keepdim=True).clamp_min(1e-8) * pos_count
        w_neg = a_neg / a_neg.sum(dim=-1, keepdim=True).clamp_min(1e-8) * neg_count

    # IPO residual: same definition as existing IPO (sequence-average log-ratio)
    tau = 1.0 / (2.0 * beta)
    delta = (policy_chosen_logps_avg - ref_chosen_logps_avg) - \
            (policy_rejected_logps_avg - ref_rejected_logps_avg)
    e = (delta - tau).detach()

    # CA-IPO surrogate margin
    delta_w = (w_pos * z_s_pos).sum(dim=-1) - (w_neg * z_s_neg).sum(dim=-1)

    loss = (e * delta_w).mean()

    metrics = {
        "ca_ipo/loss":       loss.detach(),
        "ca_ipo/delta":      delta.detach().mean(),
        "ca_ipo/e_abs":      e.abs().mean(),
        "ca_ipo/delta_w":    delta_w.detach().mean(),
        "ca_ipo/w_pos_max":  w_pos.max(dim=-1).values.mean(),
        "ca_ipo/w_neg_max":  w_neg.max(dim=-1).values.mean(),
        "ca_ipo/w_pos_mean": (w_pos * chosen_mask).sum() / chosen_mask.sum().clamp_min(1),
        "ca_ipo/w_neg_mean": (w_neg * rejected_mask).sum() / rejected_mask.sum().clamp_min(1),
        "ca_ipo/a_pos_mean": (a_pos * chosen_mask).sum() / chosen_mask.sum().clamp_min(1),
        "ca_ipo/a_neg_mean": (a_neg * rejected_mask).sum() / rejected_mask.sum().clamp_min(1),
    }
    return loss, metrics


def dpo_loss(
    chosen_logps: torch.FloatTensor,
    rejected_logps: torch.FloatTensor,
    ref_chosen_logps: torch.FloatTensor,
    ref_rejected_logps: torch.FloatTensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    reference_free: bool = False,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """
    Compute DPO loss. Ported from TRL DPOTrainer.dpo_loss.
    Returns: (losses, chosen_rewards, rejected_rewards)
    """
    device = chosen_logps.device

    chosen_logratios  = chosen_logps  - (not reference_free) * ref_chosen_logps.to(device)
    rejected_logratios = rejected_logps - (not reference_free) * ref_rejected_logps.to(device)

    if reference_free:
        ref_logratios = torch.tensor([0], dtype=chosen_logps.dtype, device=device)
    else:
        ref_logratios = ref_chosen_logps - ref_rejected_logps

    logratios = chosen_logps - rejected_logps
    logits    = logratios.to(device) - ref_logratios.to(device)

    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "robust":
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            + F.logsigmoid(-beta * logits) * label_smoothing
        ) / (1 - 2 * label_smoothing)
    elif loss_type == "hinge":
        losses = torch.relu(1 - beta * logits)
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "exo_pair":
        if label_smoothing == 0:
            label_smoothing = 1e-3
        losses = (beta * logits).sigmoid() * (
            F.logsigmoid(beta * logits) - math.log(1 - label_smoothing)
        ) + (-beta * logits).sigmoid() * (F.logsigmoid(-beta * logits) - math.log(label_smoothing))
    elif loss_type == "nca_pair":
        chosen_rewards_   = (chosen_logps  - ref_chosen_logps.to(device))  * beta
        rejected_rewards_ = (rejected_logps - ref_rejected_logps.to(device)) * beta
        losses = (
            -F.logsigmoid(chosen_rewards_)
            - 0.5 * F.logsigmoid(-chosen_rewards_)
            - 0.5 * F.logsigmoid(-rejected_rewards_)
        )
    elif loss_type == "apo_zero":
        losses = (1 - F.sigmoid(beta * chosen_logratios)) + F.sigmoid(beta * rejected_logratios)
    elif loss_type == "apo_down":
        losses = F.sigmoid(beta * chosen_logratios) + (
            1 - F.sigmoid(beta * (chosen_logratios - rejected_logratios))
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if reference_free:
        chosen_rewards   = beta * chosen_logps.detach()
        rejected_rewards = beta * rejected_logps.detach()
    else:
        chosen_rewards   = beta * (chosen_logps.to(device)  - ref_chosen_logps.to(device)).detach()
        rejected_rewards = beta * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()

    return losses, chosen_rewards, rejected_rewards


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_continuations(
    model: nn.Module,
    tokenizer,
    prompt_input_ids: torch.LongTensor,
    prompt_attention_mask: torch.LongTensor,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    """
    Generate continuations up to max_new_tokens, stopping at EOS.
    Returns: (cont_ids [B, L], cont_mask [B, L])
    where L <= max_new_tokens; tokens after EOS are padded and masked.

    FIX v2: EOS is now ALLOWED (eos_token_id restored to model default).
    Forced fixed-length generation was a bug — dense model stopping at 200
    tokens would produce garbage for the remaining 312 forced tokens.
    cont_mask correctly marks real vs padded tokens.
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    model.config.use_cache = True
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        gen_ids = model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
            # eos_token_id: use model default (EOS allowed)
        )
    model.config.use_cache = False
    prompt_len = prompt_input_ids.shape[1]
    cont_ids = gen_ids[:, prompt_len:]  # [B, L]

    # Build attention mask: 1 up to and including EOS, 0 for padding after.
    #
    # FIX: DeepSeek/Qwen tokenizers have pad_token_id == eos_token_id.
    # In that case, generated padding tokens are indistinguishable from EOS
    # by token-id alone. We rely solely on the cumsum trick:
    #   after_eos[i] = cumsum of (token == eos_id) → 0 before 1st EOS,
    #   1 at 1st EOS position, >=2 for all pad tokens after.
    # So (after_eos <= 1) correctly marks real tokens + 1st EOS as valid,
    # and masks all subsequent pad/EOS tokens — even when pad == eos.
    #
    # We only apply the extra (cont_ids != pad_id) filter when pad ≠ eos,
    # i.e. when there are genuine pad tokens that carry no EOS semantics.
    if eos_id is not None:
        is_eos    = (cont_ids == eos_id)
        after_eos = is_eos.long().cumsum(dim=1)   # 0…0, 1, ≥2 after first EOS
        cont_mask = (after_eos <= 1).long()        # real tokens + 1st EOS = 1
        if pad_id != eos_id:
            # genuine padding is distinguishable; mask it out too
            cont_mask = cont_mask * (cont_ids != pad_id).long()
    else:
        cont_mask = (cont_ids != pad_id).long()

    return cont_ids, cont_mask


def _chosen_cache_key(prompt_path: str, n_pairs: int, max_new_tokens: int, temperature: float) -> str:
    """Deterministic hash key for chosen cache based on generation params."""
    raw = f"{prompt_path}|{n_pairs}|{max_new_tokens}|{temperature:.4f}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def generate_chosen_cache(
    dense_model: nn.Module,
    tokenizer,
    prompt_dataset,
    n_pairs: int,
    gen_batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    device: str = "cuda",
    cache_dir: str | None = None,
    prompt_path: str = "",
    store_teacher_logps: bool = False,
) -> list[dict]:
    """
    Pre-generate chosen (dense) continuations for n_pairs prompts.
    Returns list of dicts: {prompt_input_ids, prompt_attention_mask,
                             chosen_input_ids, chosen_attention_mask}

    If cache_dir is given, saves/loads cache from disk keyed by generation params.
    Cache is invalidated automatically when n_pairs/max_new_tokens/temperature/prompt_path change.
    """
    from lib.gkd_admm_trainer import collate_prompts

    if cache_dir:
        key = _chosen_cache_key(prompt_path, n_pairs, max_new_tokens, temperature)
        cache_file = os.path.join(cache_dir, f"chosen_cache_{key}.pt")
        if os.path.exists(cache_file):
            logging.info(f"Loading chosen cache from disk: {cache_file}")
            return torch.load(cache_file, map_location="cpu")

    loader = DataLoader(
        prompt_dataset,
        batch_size=gen_batch_size,
        shuffle=False,
        collate_fn=collate_prompts(tokenizer.pad_token_id or 0),
    )

    cache = []
    dense_model.eval()
    logging.info(f"Generating chosen cache: {n_pairs} pairs (batch={gen_batch_size}) ...")
    for batch in loader:
        if len(cache) >= n_pairs:
            break
        prompt_ids  = batch["input_ids"].to(device)
        prompt_mask = batch["attention_mask"].to(device)

        cont_ids, cont_mask = generate_continuations(
            dense_model, tokenizer, prompt_ids, prompt_mask,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )

        # Optionally compute teacher (dense model) token logps for CA-IPO
        if store_teacher_logps:
            teacher_tok_logps = get_completion_token_logps(
                dense_model,
                prompt_ids, prompt_mask,
                cont_ids.to(device), cont_mask.to(device),
            ).cpu()

        for i in range(prompt_ids.shape[0]):
            if len(cache) >= n_pairs:
                break
            entry = {
                "prompt_input_ids":       prompt_ids[i:i+1].cpu(),
                "prompt_attention_mask":  prompt_mask[i:i+1].cpu(),
                "chosen_input_ids":       cont_ids[i:i+1].cpu(),
                "chosen_attention_mask":  cont_mask[i:i+1].cpu(),
            }
            if store_teacher_logps:
                entry["teacher_chosen_token_logps"] = teacher_tok_logps[i:i+1]
            cache.append(entry)
    logging.info(f"Chosen cache ready: {len(cache)} pairs")

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        torch.save(cache, cache_file)
        logging.info(f"Chosen cache saved to disk: {cache_file}")

    return cache


# ---------------------------------------------------------------------------
# Rejected queue
# ---------------------------------------------------------------------------

class RejectedQueue:
    """
    Maintains a queue of rejected continuations regenerated every mask_interval steps.

    Usage:
        queue = RejectedQueue(chosen_cache, mask_interval, ...)
        # after each mask update:
        queue.refill(sparse_model, tokenizer, device)
        # each training step:
        pair = queue.pop()  # returns None if empty
    """

    def __init__(
        self,
        chosen_cache: list[dict],
        mask_interval: int,
        gen_batch_size: int = 8,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        teacher_model: nn.Module | None = None,
    ):
        self.chosen_cache   = chosen_cache
        self.mask_interval  = mask_interval
        self.gen_batch_size = gen_batch_size
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self.teacher_model  = teacher_model  # for CA-IPO: compute teacher logps on rejected
        self._queue: deque  = deque()
        self._ptr: int      = 0   # pointer into chosen_cache

    def refill(self, sparse_model: nn.Module, tokenizer, device: str):
        """
        Generate mask_interval rejected continuations from sparse_model
        and push (chosen_entry, rejected_cont) pairs into the queue.
        """
        n = min(self.mask_interval, len(self.chosen_cache))
        idxs = [(self._ptr + i) % len(self.chosen_cache) for i in range(n)]
        self._ptr = (self._ptr + n) % len(self.chosen_cache)

        sparse_model.eval()
        # batch-generate for efficiency
        all_prompt_ids, all_prompt_masks = [], []
        for idx in idxs:
            all_prompt_ids.append(self.chosen_cache[idx]["prompt_input_ids"])
            all_prompt_masks.append(self.chosen_cache[idx]["prompt_attention_mask"])

        all_cont_ids, all_cont_masks = [], []
        for start in range(0, len(idxs), self.gen_batch_size):
            end   = start + self.gen_batch_size
            p_ids = torch.cat(all_prompt_ids[start:end], dim=0).to(device)
            p_msk = torch.cat(all_prompt_masks[start:end], dim=0).to(device)
            cont_ids, cont_mask = generate_continuations(
                sparse_model, tokenizer, p_ids, p_msk,
                max_new_tokens=self.max_new_tokens, temperature=self.temperature,
            )
            all_cont_ids.append(cont_ids.cpu())
            all_cont_masks.append(cont_mask.cpu())

        # pad all batches to the same continuation length for stacking
        max_cont_len = max(c.shape[1] for c in all_cont_ids)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        padded_ids   = [pad_to_length(c, max_cont_len, pad_id)   for c in all_cont_ids]
        padded_masks = [pad_to_length(m, max_cont_len, 0)        for m in all_cont_masks]
        results_ids  = torch.cat(padded_ids,   dim=0)  # [n, max_cont_len]
        results_mask = torch.cat(padded_masks, dim=0)  # [n, max_cont_len]

        # If teacher_model provided (CA-IPO), compute teacher token logps on rejected
        teacher_rejected_logps = None
        if self.teacher_model is not None:
            # Batch all (prompt, rejected) pairs for teacher forward
            teacher_logps_list = []
            for start in range(0, len(idxs), self.gen_batch_size):
                end = start + self.gen_batch_size
                batch_idxs = idxs[start:end]
                p_ids  = torch.cat([self.chosen_cache[i]["prompt_input_ids"]  for i in batch_idxs], dim=0).to(device)
                p_msk  = torch.cat([self.chosen_cache[i]["prompt_attention_mask"] for i in batch_idxs], dim=0).to(device)
                r_ids  = results_ids[start:end].to(device)
                r_msk  = results_mask[start:end].to(device)
                # Pad rejected to same length within this sub-batch (already same via stacking above)
                t_logps = get_completion_token_logps(
                    self.teacher_model, p_ids, p_msk, r_ids, r_msk,
                )
                teacher_logps_list.append(t_logps.cpu())
            teacher_rejected_logps = torch.cat(teacher_logps_list, dim=0)  # [n, max_cont_len]

        for local_i, global_idx in enumerate(idxs):
            entry = {
                **self.chosen_cache[global_idx],
                "rejected_input_ids":       results_ids[local_i:local_i+1],
                "rejected_attention_mask":  results_mask[local_i:local_i+1],
            }
            if teacher_rejected_logps is not None:
                entry["teacher_rejected_token_logps"] = teacher_rejected_logps[local_i:local_i+1]
            self._queue.append(entry)
        logging.info(f"RejectedQueue refilled: {len(self._queue)} pairs ready")

    def pop(self) -> dict | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def peek_n(self, n: int) -> list[dict]:
        """Return up to n pairs from the front of the queue without removing them."""
        return [self._queue[i] for i in range(min(n, len(self._queue)))]

    def __len__(self):
        return len(self._queue)


@torch.no_grad()
def compute_teacher_delta(
    teacher_model: nn.Module,
    pairs: list[dict],
    padding_value: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute Δ_T = avg_log_p_T(chosen|x) - avg_log_p_T(rejected|x) for each pair.
    Returns tensor of shape [N] (one value per pair).

    Δ_T > 0 means teacher prefers chosen over rejected → valid preference pair.
    Δ_T ≈ 0 means teacher can't distinguish → noisy pair (early pruning).
    """
    teacher_model.eval()
    deltas = []
    for pair in pairs:
        pair_dev = {k: v.to(device) for k, v in pair.items()}
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = concatenated_forward(
                teacher_model, pair_dev,
                padding_value=padding_value,
                is_ref_model=True,
            )
        delta = out["chosen_logps"] - out["rejected_logps"]  # [1]
        deltas.append(delta.cpu())
    return torch.cat(deltas, dim=0)  # [N]
